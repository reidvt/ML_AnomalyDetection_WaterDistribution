"""
config.py
─────────
Central configuration for the LeakDB Spatio-Temporal Leak Detection Pipeline.

Paths are expressed as raw strings so they work on both Windows (during dev)
and Linux (AI Lab).  The __post_init__ resolves them to absolute form so the
rest of the code never has to think about ~ or relative paths.

Key scaling defaults for 1000+ scenarios on the AI Lab:
  - num_workers is auto-detected from CPU count (Linux only; 0 on Windows)
  - pin_memory is gated on CUDA availability
  - cache_size (passed to LeakWindowDataset) is separate from this config
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch


# ── OS / hardware detection (runs once at import) ────────────────────────────
_ON_WINDOWS = sys.platform.startswith("win")
_CUDA_AVAIL  = torch.cuda.is_available()

# Safe num_workers default:
#   Windows  → 0  (spawn-based multiprocessing kills lru_cache / lambdas)
#   Linux    → half the logical CPU count, capped at 12
if _ON_WINDOWS:
    _DEFAULT_WORKERS = 0
else:
    _DEFAULT_WORKERS = min(max(os.cpu_count() // 2, 1), 12)


@dataclass
class PipelineConfig:

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Edit these for your machine.  Raw strings (r"...") work on both platforms.

    # Directory of Parquet master files produced by preprocess.py
    data_root: str = r"C:\Users\reidv\SDX\SEM2\MergedTraining\data_master"

    # EPANET .inp file — set to None to fall back to heuristic ring topology
    inp_file: Optional[str] = r"C:\Users\reidv\SDX\SEM2\MergedTraining\Hanoi_CMH.inp"

    # Output root; subdirs (models/, embeddings/, logs/) are created automatically
    output_dir: str = r"C:\Users\reidv\SDX\SEM2\MergedTraining\outputs"

    # ── Data Windowing ────────────────────────────────────────────────────────
    window_size: int = 48           # timesteps per window
    stride:      int = 24           # hop between windows
    max_scenarios: Optional[int] = None   # None = all files

    # ── Feature selection ────────────────────────────────────────────────────
    use_pressure: bool = True
    use_demand:   bool = False
    use_flows:    bool = True

    # ── Dataset cache ─────────────────────────────────────────────────────────
    # Number of Parquet DataFrames held in RAM across all workers.
    # 1000 scenarios × ~50 MB each = serious RAM if set too high.
    # 64 is safe on a 128 GB AI Lab machine; drop to 16-32 on 32 GB.
    dataset_cache_size: int = 64

    # ── Splits ───────────────────────────────────────────────────────────────
    val_split:  float = 0.15
    test_split: float = 0.15

    # ── Temporal encoder (LSTM) ──────────────────────────────────────────────
    lstm_hidden:   int   = 64
    lstm_layers:   int   = 2
    lstm_dropout:  float = 0.2
    embedding_dim: int   = 32

    # ── Spatial GNN ───────────────────────────────────────────────────────────
    gnn_hidden:    int   = 64
    gnn_layers:    int   = 2
    gnn_dropout:   float = 0.2
    edge_feat_dim: int   = 1

    # ── Training ─────────────────────────────────────────────────────────────
    epochs:       int   = 150
    batch_size:   int   = 32        # increased from 16 — AI Lab GPU can handle it
    lr:           float = 1e-3
    weight_decay: float = 1e-5
    pos_weight:   float = 10.0
    grad_clip:    float = 1.0

    # Auto-detected; override here if needed (e.g. num_workers=8)
    num_workers: int  = _DEFAULT_WORKERS

    # Only True when CUDA is actually present — prevents DataLoader hangs
    pin_memory:  bool = _CUDA_AVAIL

    # ── Early stopping ────────────────────────────────────────────────────────
    patience:   int   = 15
    min_delta:  float = 0.001

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Derived (auto-set) ────────────────────────────────────────────────────
    input_dim: int = field(init=False)

    def __post_init__(self):
        # ── input_dim ─────────────────────────────────────────────────────
        self.input_dim = int(self.use_pressure) + int(self.use_demand)
        if self.input_dim == 0:
            raise ValueError(
                "At least one of `use_pressure` or `use_demand` must be True."
            )

        # ── Resolve all paths to absolute form ────────────────────────────
        self.data_root  = os.path.abspath(os.path.expanduser(self.data_root))
        self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        if self.inp_file:
            self.inp_file = os.path.abspath(os.path.expanduser(self.inp_file))
            if not os.path.exists(self.inp_file):
                print(
                    f"[Config] INP file not found at '{self.inp_file}'. "
                    "Will fall back to heuristic topology."
                )
                self.inp_file = None

        # ── Create output subdirectories ──────────────────────────────────
        for sub in ("models", "embeddings", "logs"):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

        # ── Print effective settings ──────────────────────────────────────
        print(
            f"[Config] OS={'Windows' if _ON_WINDOWS else 'Linux'}  "
            f"CUDA={_CUDA_AVAIL}  "
            f"num_workers={self.num_workers}  "
            f"pin_memory={self.pin_memory}  "
            f"batch_size={self.batch_size}"
        )
        print(f"[Config] data_root  = {self.data_root}")
        print(f"[Config] output_dir = {self.output_dir}")
