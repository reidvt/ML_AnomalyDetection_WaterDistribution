"""
train.py  ── hardened for multi-day runs + DDP (DistributedDataParallel)
─────────────────────────────────────────────────────────────────────────
Multi-GPU support
─────────────────
  The previous nn.DataParallel approach is REMOVED. DataParallel breaks
  PyG Batch objects because it scatters/gathers standard tensors but does
  not understand graph-aware attributes (batch.batch, batch.edge_index,
  batch.edge_attr).  PyG Batch objects must travel whole to each GPU.

  DDP fix: torchrun launches one process per GPU. Each process owns one
  GPU. Gradients are averaged across GPUs via NCCL all-reduce after every
  backward pass. The model stays identical on both GPUs at all times, so
  validation metrics computed independently on each rank are identical —
  no gather step needed.

Launch (2 GPUs, pw3, 150 epochs, patience 120):
    torchrun --nproc_per_node=2 train.py \\
        --output_dir  /home/ailab/Desktop/LeakDB_SDX/outputs/full_pw3_ddp \\
        --data_root   /home/ailab/Desktop/LeakDB_SDX/LeakDB/processeddata \\
        --pos_weight  3 \\
        --lr          0.0003 \\
        --batch_size  200 \\
        --grad_clip   0.5 \\
        --epochs      150 \\
        --patience    120 \\
        --num_workers 12 \\
        --dataset_cache_size 128

Launch (single GPU — identical command, just no torchrun):
    python train.py --epochs 150 --patience 120 ...

Resume (DDP):
    torchrun --nproc_per_node=2 train.py --resume ...

Key DDP design decisions
─────────────────────────
  DistributedSampler   train set only — ensures each GPU sees different
                       samples within an epoch (no duplicate gradients).
                       Val / test run on both ranks with full loader; since
                       DDP keeps weights in sync results are identical, so
                       rank 0 simply logs without needing to gather.

  Rank-0 gate          All file I/O (logging, checkpointing, heartbeat,
                       early-stop model save) happens only on rank 0. Rank 1
                       participates in training and val but is silent.

  Early-stop broadcast After rank 0 decides to stop, it broadcasts a 1-byte
                       tensor so rank 1 breaks out of the loop at the same
                       epoch. Without this, rank 1 would hang at the next
                       dist.barrier() inside run_epoch waiting for rank 0.

  Two barriers per epoch
                       1) After training, before val   — ensures rank 1 does
                          not start the next train epoch while rank 0 is still
                          running val (and vice-versa in the other direction).
                       2) After val, before stop-broadcast — fixes the NCCL
                          timeout bug documented in CLAUDE.md where rank 1
                          arrived at the broadcast 10 min early → SIGABRT.

  num_workers per rank With torchrun each process spawns its own workers.
                       If cfg.num_workers=12 and nproc_per_node=2 you get
                       24 worker processes total. Halve if RAM is tight.
"""

import argparse
import csv
import os
import signal
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import PipelineConfig
from dataset import LeakWindowDataset
from models import SpatioTemporalLeakDetector
from utils import EarlyStoppingF1, compute_metrics, save_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────────────

def _init_distributed() -> Tuple[int, int]:
    """
    Initialize the NCCL process group (called only under torchrun).
    Returns (local_rank, world_size).
    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE in the environment.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def _broadcast_stop(stop: bool, device: torch.device) -> bool:
    """
    Rank 0 broadcasts its early-stop decision to all ranks.
    Returns True if training should stop.
    """
    t = torch.tensor([int(stop)], dtype=torch.int32, device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeakDB Spatio-Temporal Trainer")
    p.add_argument("--data_root",          type=str,   default=None)
    p.add_argument("--inp_file",           type=str,   default=None)
    p.add_argument("--output_dir",         type=str,   default=None)
    p.add_argument("--max_scenarios",      type=int,   default=None)
    p.add_argument("--window_size",        type=int,   default=None)
    p.add_argument("--stride",             type=int,   default=None)
    p.add_argument("--batch_size",         type=int,   default=None)
    p.add_argument("--epochs",             type=int,   default=None)
    p.add_argument("--lr",                 type=float, default=None)
    p.add_argument("--patience",           type=int,   default=None)
    p.add_argument("--pos_weight",         type=float, default=None,
                   help="BCEWithLogitsLoss positive class weight (default 10.0)")
    p.add_argument("--grad_clip",          type=float, default=None,
                   help="Gradient clip max norm (default 1.0; use 0.5 for "
                        "large batch or high pos_weight)")
    p.add_argument("--lstm_hidden",        type=int,   default=None)
    p.add_argument("--lstm_dropout",       type=float, default=None)
    p.add_argument("--embedding_dim",      type=int,   default=None)
    p.add_argument("--gnn_hidden",         type=int,   default=None)
    p.add_argument("--gnn_layers",         type=int,   default=None)
    p.add_argument("--gnn_dropout",        type=float, default=None)
    p.add_argument("--num_workers",        type=int,   default=None)
    p.add_argument("--dataset_cache_size", type=int,   default=None)
    p.add_argument("--checkpoint_every",   type=int,   default=5,
                   help="Save a milestone checkpoint every N epochs (default 5)")
    p.add_argument("--use_demand",   action="store_true", default=None)
    p.add_argument("--no_flows",     action="store_true")
    p.add_argument("--resume",       action="store_true",
                   help="Resume training from checkpoint_latest.pth")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Directories
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dirs(output_dir: str) -> dict:
    dirs = {
        "root":       output_dir,
        "models":     os.path.join(output_dir, "models"),
        "embeddings": os.path.join(output_dir, "embeddings"),
        "logs":       os.path.join(output_dir, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# Epoch logger  (appends on resume)
# ──────────────────────────────────────────────────────────────────────────────

class EpochLogger:
    FIELDS = ["epoch", "train_loss", "val_loss", "val_f1",
              "val_precision", "val_recall", "val_roc_auc", "elapsed_s"]

    def __init__(self, path: str, resume: bool = False):
        self.path = path
        if not resume or not os.path.exists(path):
            with open(path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.FIELDS).writeheader()
            print(f"  📝 Epoch log (new)    → {path}")
        else:
            print(f"  📝 Epoch log (append) → {path}")

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.FIELDS,
                           extrasaction="ignore").writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# Heartbeat
# ──────────────────────────────────────────────────────────────────────────────

def _write_heartbeat(path: str, epoch: int, best_f1: float,
                     world_size: int = 1) -> None:
    with open(path, "w") as fh:
        fh.write(
            f"alive\n"
            f"epoch={epoch}\n"
            f"best_val_f1={best_f1:.4f}\n"
            f"gpus={world_size}\n"
            f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Worker init
# ──────────────────────────────────────────────────────────────────────────────

def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ──────────────────────────────────────────────────────────────────────────────

def _save_latest(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    early_stopper: EarlyStoppingF1,
    epoch: int,
    val_metrics: dict,
    elapsed_so_far: float,
) -> None:
    """
    Atomic overwrite of checkpoint_latest.pth.  Only called from rank 0.
    Unwraps DDP via model.module so the checkpoint loads on any GPU count.
    """
    raw_model = model.module if isinstance(model, DDP) else model
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "stopper_best_f1": early_stopper.best_f1,
            "stopper_counter": early_stopper.counter,
            "stopper_stopped": early_stopper.early_stop,
            "val_metrics":     val_metrics,
            "elapsed_s":       elapsed_so_far,
        },
        tmp,
    )
    os.replace(tmp, path)


def _load_latest(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    early_stopper: EarlyStoppingF1,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Restore from checkpoint_latest.pth.
    Works regardless of whether the model is DDP-wrapped or not.
    Returns (start_epoch, elapsed_seconds_already_spent).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Unwrap DDP before loading state dict
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state"])

    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])

    early_stopper.best_f1    = ckpt["stopper_best_f1"]
    early_stopper.counter    = ckpt["stopper_counter"]
    early_stopper.early_stop = ckpt["stopper_stopped"]

    start_epoch  = ckpt["epoch"] + 1
    elapsed_prev = ckpt.get("elapsed_s", 0.0)

    print(
        f"  ✅ Resumed from epoch {ckpt['epoch']}  |  "
        f"Best F1 so far: {early_stopper.best_f1:.4f}  |  "
        f"Time already spent: {elapsed_prev/3600:.2f} h"
    )
    return start_epoch, elapsed_prev


# ──────────────────────────────────────────────────────────────────────────────
# Single epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, device, scaler,
    optimizer=None, is_train: bool = True, desc: str = "",
    grad_clip: float = 1.0,
    show_progress: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run one pass through the dataloader.

    show_progress=False on rank > 0 to avoid duplicate tqdm bars.
    All ranks participate in forward/backward; only rank 0 shows output.
    """
    model.train() if is_train else model.eval()
    total_loss    = 0.0
    valid_batches = 0
    nan_batches   = 0
    all_probs:  list = []
    all_labels: list = []

    pbar = tqdm(
        loader,
        desc=f"  {desc:6s}",
        leave=False,
        unit="batch",
        disable=not show_progress,
    )

    for batch in pbar:
        batch = batch.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda",
                                    enabled=(device.type == "cuda")):
                _, graph_logits = model(batch)
                y    = (batch.y > 0).float().to(device)
                loss = criterion(graph_logits, y)

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            nan_batches += 1
            if show_progress and (nan_batches <= 5 or nan_batches % 50 == 0):
                print(
                    f"\n  ⚠️  NaN/Inf loss ({nan_batches} so far). "
                    "Skipping update. Reduce --lr or --grad_clip."
                )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            probs = torch.full(
                (graph_logits.shape[0],), 0.5, device="cpu", dtype=torch.float32
            )
            all_probs.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_val
        valid_batches += 1
        all_probs.extend(
            torch.sigmoid(graph_logits.detach()).cpu().numpy().tolist()
        )
        all_labels.extend(y.cpu().numpy().tolist())
        if show_progress:
            pbar.set_postfix(loss=f"{loss_val:.4f}")

    if show_progress and nan_batches > 0:
        print(f"\n  ⚠️  Epoch had {nan_batches} NaN/Inf batches "
              f"out of {len(loader)} total.")

    avg_loss = total_loss / max(valid_batches, 1)
    return (
        avg_loss,
        np.array(all_probs,  dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Embedding export
# ──────────────────────────────────────────────────────────────────────────────

def export_embeddings(model, dataset, cfg, device, emb_dir: str) -> None:
    print("\n📦 Exporting node embeddings…")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
    )
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    scenario_store: dict = {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="  embed", leave=False)):
            batch      = batch.to(device, non_blocking=True)
            embeddings = raw_model.encode_nodes(batch)
            for g in batch.batch.unique():
                mask     = batch.batch == g
                emb_np   = embeddings[mask].cpu().numpy()
                flat_idx = i * cfg.batch_size + int(g.item())
                if flat_idx >= len(dataset._windows):
                    continue
                w = dataset._windows[flat_idx]
                scenario_store.setdefault(w["pid"], []).append(
                    (w["start"], emb_np)
                )

    for pid, windows in scenario_store.items():
        windows.sort(key=lambda t: t[0])
        arr = np.array([w[1] for w in windows])
        np.save(os.path.join(emb_dir, f"emb_{pid}.npy"), arr)

    print(f"  ✅ Embeddings for {len(scenario_store)} scenarios → {emb_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────────
    # torchrun injects LOCAL_RANK into the environment.
    # If it is absent we are running single-GPU (python train.py ...).
    is_distributed = "LOCAL_RANK" in os.environ
    local_rank  = 0
    world_size  = 1
    is_rank0    = True

    if is_distributed:
        local_rank, world_size = _init_distributed()
        is_rank0 = (local_rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")

    # ── Config ────────────────────────────────────────────────────────────────
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    overrides.pop("resume", None)
    checkpoint_every: int = overrides.pop("checkpoint_every", 5)
    if overrides.pop("no_flows", False):
        overrides["use_flows"] = False

    cfg = PipelineConfig(**{
        k: v for k, v in overrides.items()
        if k in PipelineConfig.__dataclass_fields__
    })

    # Only rank 0 creates directories (avoids race on NFS / shared FS)
    if is_rank0:
        dirs = _ensure_dirs(cfg.output_dir)
        print("\n📁 Output paths:")
        for name, path in dirs.items():
            print(f"   {name+'/':12s} {path}")
    else:
        # Rank > 0 still needs the dict for path construction later
        dirs = {
            "root":       cfg.output_dir,
            "models":     os.path.join(cfg.output_dir, "models"),
            "embeddings": os.path.join(cfg.output_dir, "embeddings"),
            "logs":       os.path.join(cfg.output_dir, "logs"),
        }

    # Barrier: wait until rank 0 has created dirs before any rank opens files
    if is_distributed:
        dist.barrier()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed + local_rank)
    np.random.seed(cfg.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + local_rank)

    if is_rank0:
        print(f"\n🖥️  Device : {device}  |  world_size={world_size}")
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(local_rank)
            print(f"     GPU   : {torch.cuda.get_device_name(local_rank)}")
            print(f"     VRAM  : {props.total_memory / 1e9:.1f} GB")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if is_rank0:
        print(f"\n📂 Building dataset from {cfg.data_root} …")

    cache_size   = getattr(cfg, "dataset_cache_size", 64)
    full_dataset = LeakWindowDataset(cfg, cache_size=cache_size)

    n       = len(full_dataset)
    n_test  = max(1, int(cfg.test_split * n))
    n_val   = max(1, int(cfg.val_split  * n))
    n_train = n - n_val - n_test

    if is_rank0:
        print(f"  Split → Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    if is_rank0:
        np.save(
            os.path.join(dirs["logs"], "test_indices.npy"),
            np.array(test_set.indices),
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    use_persistent = cfg.num_workers > 0
    prefetch       = 2 if cfg.num_workers > 0 else None

    loader_kw = dict(
        num_workers        = cfg.num_workers,
        pin_memory         = cfg.pin_memory,
        worker_init_fn     = _worker_init_fn if cfg.num_workers > 0 else None,
        persistent_workers = use_persistent,
        prefetch_factor    = prefetch,
    )

    # Training: DistributedSampler ensures no two GPUs see the same samples
    # in the same epoch. shuffle=True is handled by the sampler, not the loader.
    if is_distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=local_rank,
            shuffle=True, seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size,
            sampler=train_sampler, shuffle=False, **loader_kw
        )
    else:
        train_sampler = None
        train_loader  = DataLoader(
            train_set, batch_size=cfg.batch_size,
            shuffle=True, **loader_kw
        )

    # Val / test: both ranks run the FULL loader.
    # Since DDP keeps weights identical across ranks, metrics are identical.
    # Rank 0 logs; rank 1 computes but stays silent. No gather needed.
    val_loader  = DataLoader(
        val_set,  batch_size=cfg.batch_size, shuffle=False, **loader_kw
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, **loader_kw
    )

    if is_rank0:
        print(
            f"  DataLoader: workers={cfg.num_workers}  "
            f"persistent={use_persistent}  prefetch={prefetch}  "
            f"pin_memory={cfg.pin_memory}  "
            f"DDP={is_distributed} (world_size={world_size})"
        )

    # ── Model ────────────────────────────────────────────────────────────────
    model = SpatioTemporalLeakDetector(cfg).to(device)

    if is_distributed:
        # find_unused_parameters=False is safe here: every parameter
        # participates in every forward pass.
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False)

    if is_rank0:
        raw = model.module if isinstance(model, DDP) else model
        print(f"\n🧠 Model : {raw.count_parameters():,} trainable parameters")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, verbose=is_rank0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.pos_weight], device=device)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── File paths ────────────────────────────────────────────────────────────
    best_weights_path = os.path.join(dirs["models"], "best_model.pth")
    latest_ckpt_path  = os.path.join(dirs["models"], "checkpoint_latest.pth")
    log_csv_path      = os.path.join(dirs["logs"],   "training_log.csv")
    metrics_txt_path  = os.path.join(dirs["logs"],   "final_metrics.txt")
    heartbeat_path    = os.path.join(dirs["logs"],   "heartbeat.txt")

    if is_rank0:
        print(f"\n  Best weights   → {best_weights_path}")
        print(f"  Latest ckpt    → {latest_ckpt_path}")

    early_stopper = EarlyStoppingF1(
        patience=cfg.patience, min_delta=cfg.min_delta, path=best_weights_path
    )
    epoch_logger = EpochLogger(log_csv_path, resume=args.resume) if is_rank0 else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 1
    elapsed_prev = 0.0

    if args.resume:
        if os.path.exists(latest_ckpt_path):
            if is_rank0:
                print(f"\n🔄 Resuming from {latest_ckpt_path} …")
            start_epoch, elapsed_prev = _load_latest(
                latest_ckpt_path, model, optimizer, scheduler,
                scaler, early_stopper, device,
            )
        else:
            if is_rank0:
                print(
                    f"\n⚠️  --resume passed but {latest_ckpt_path} not found. "
                    "Starting from epoch 1."
                )

    # Sync start_epoch across ranks (rank 0 may have loaded a later epoch)
    if is_distributed:
        t = torch.tensor([start_epoch], dtype=torch.int32, device=device)
        dist.broadcast(t, src=0)
        start_epoch = int(t.item())

    # ── Signal handler (SIGTERM / Ctrl-C) ─────────────────────────────────────
    # Only rank 0 writes the checkpoint; rank 1 just exits cleanly.
    _state: dict = {}

    def _handle_signal(signum, frame):
        if is_rank0:
            print(f"\n⚡ Signal {signum} received — saving checkpoint…")
            if _state.get("model") is not None:
                _save_latest(
                    latest_ckpt_path,
                    _state["model"], _state["optimizer"],
                    _state["scheduler"], _state["scaler"],
                    _state["early_stopper"],
                    _state.get("epoch", start_epoch - 1),
                    _state.get("val_metrics", {}),
                    time.time() - _state["t0"] + elapsed_prev,
                )
                print("  ✅ Checkpoint saved. Exiting.")
        if is_distributed:
            dist.destroy_process_group()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    # ── Training loop ─────────────────────────────────────────────────────────
    if is_rank0:
        print(
            f"\n🚀 Training  (epochs {start_epoch}–{cfg.epochs}  |  "
            f"patience={cfg.patience}  |  "
            f"milestone every {checkpoint_every} epochs  |  "
            f"GPUs={world_size})\n"
        )

    t0 = time.time()
    _state.update({
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "scaler": scaler, "early_stopper": early_stopper, "t0": t0,
    })

    for epoch in range(start_epoch, cfg.epochs + 1):

        # Tell DistributedSampler which epoch this is so it shuffles differently
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Train (all ranks) ────────────────────────────────────────────────
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, scaler,
            optimizer=optimizer, is_train=True, desc="train",
            grad_clip=cfg.grad_clip,
            show_progress=is_rank0,
        )

        # Barrier 1: all ranks finish training before any starts val.
        # Without this, rank 1 could start the next epoch while rank 0
        # is still running val, causing a DDP deadlock.
        if is_distributed:
            dist.barrier()

        # ── Val (all ranks — results are identical, rank 0 logs) ─────────────
        val_loss, val_probs, val_labels = run_epoch(
            model, val_loader, criterion, device, scaler,
            optimizer=None, is_train=False, desc="val",
            grad_clip=cfg.grad_clip,
            show_progress=is_rank0,
        )

        # Barrier 2: all ranks finish val before rank 0 broadcasts stop signal.
        # This is the fix for the NCCL timeout / SIGABRT bug documented in
        # CLAUDE.md — rank 1 was arriving at broadcast before rank 0 started it.
        if is_distributed:
            dist.barrier()

        # ── Metrics, logging, checkpointing (rank 0 only) ────────────────────
        should_stop = False

        if is_rank0:
            val_m   = compute_metrics(val_labels, val_probs)
            elapsed = time.time() - t0 + elapsed_prev
            scheduler.step(val_m["f1"])

            _state["epoch"]       = epoch
            _state["val_metrics"] = val_m

            print(
                f"Epoch {epoch:03d}/{cfg.epochs}  |  "
                f"TrLoss {train_loss:.4f}  ValLoss {val_loss:.4f}  |  "
                f"F1 {val_m['f1']:.4f}  AUC {val_m['roc_auc']:.4f}  "
                f"P {val_m['precision']:.4f}  R {val_m['recall']:.4f}  |  "
                f"{elapsed/3600:.2f}h"
            )

            epoch_logger.log({
                "epoch":         epoch,
                "train_loss":    round(train_loss, 6),
                "val_loss":      round(val_loss, 6),
                "val_f1":        round(val_m["f1"], 6),
                "val_precision": round(val_m["precision"], 6),
                "val_recall":    round(val_m["recall"], 6),
                "val_roc_auc":   round(val_m["roc_auc"], 6),
                "elapsed_s":     round(elapsed, 1),
            })

            _write_heartbeat(heartbeat_path, epoch, early_stopper.best_f1,
                             world_size=world_size)
            _save_latest(
                latest_ckpt_path, model, optimizer, scheduler,
                scaler, early_stopper, epoch, val_m, elapsed,
            )

            if epoch % checkpoint_every == 0:
                ckpt = os.path.join(
                    dirs["models"], f"checkpoint_ep{epoch:04d}.pth"
                )
                save_checkpoint(ckpt, model, optimizer, epoch, val_m)
                print(f"  💾 Milestone → {ckpt}")

            early_stopper(val_m["f1"], model)
            if early_stopper.early_stop:
                print(
                    f"\n🛑 Auto-stop at epoch {epoch}.  "
                    f"Best Val F1 = {early_stopper.best_f1:.4f}"
                )
                should_stop = True

        # ── Broadcast early-stop decision from rank 0 to all ranks ───────────
        if is_distributed:
            should_stop = _broadcast_stop(should_stop, device)

        if should_stop:
            break

    # ── Test evaluation (rank 0 only) ─────────────────────────────────────────
    if is_rank0:
        total = time.time() - t0 + elapsed_prev
        print(f"\n💎 Loading best weights ({early_stopper.best_f1:.4f} F1) …")

        # Load best weights into the raw (non-DDP) model for inference
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(
            torch.load(best_weights_path, map_location=device, weights_only=True)
        )

        _, test_probs, test_labels = run_epoch(
            model, test_loader, criterion, device, scaler,
            optimizer=None, is_train=False, desc="test",
            grad_clip=cfg.grad_clip,
            show_progress=True,
        )
        np.save(os.path.join(dirs["logs"], "test_probs.npy"),  test_probs)
        np.save(os.path.join(dirs["logs"], "test_labels.npy"), test_labels)

        test_m = compute_metrics(test_labels, test_probs)
        lines = [
            "=" * 60,
            "  FINAL TEST SET EVALUATION",
            "=" * 60,
            f"  {'F1':14s} : {test_m['f1']:.4f}",
            f"  {'PRECISION':14s} : {test_m['precision']:.4f}",
            f"  {'RECALL':14s} : {test_m['recall']:.4f}",
            f"  {'ROC-AUC':14s} : {test_m['roc_auc']:.4f}",
            "=" * 60,
            f"  Best val F1    : {early_stopper.best_f1:.4f}",
            f"  Total runtime  : {total/3600:.2f} h",
            f"  GPUs used      : {world_size}",
            "=" * 60,
        ]
        for line in lines:
            print(line)
        with open(metrics_txt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n  Metrics → {metrics_txt_path}")

        try:
            from evaluate import plot_all as _plot_all
            exp_name = os.path.basename(cfg.output_dir)
            print(f"\n📊 Generating plots for {exp_name} …")
            _plot_all(cfg.output_dir, threshold=0.5, experiment_name=exp_name)
        except Exception as plot_err:
            print(f"\n  ⚠️  Plot generation failed: {plot_err}")
            print(f"  Run manually: python evaluate.py --output_dir {dirs['root']}")

        export_embeddings(model, full_dataset, cfg, device, dirs["embeddings"])
        print(f"\n🎉 Complete.  Total runtime: {total/3600:.2f} h")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if is_distributed:
        dist.destroy_process_group()


# ── Entry-point guard ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
