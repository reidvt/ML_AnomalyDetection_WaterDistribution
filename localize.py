"""
localize.py
───────────
Leak *localization* using per-node GINEConv logits.

Unlike evaluate.py (which only measures graph-level detection), this script
uses the node_logits [total_nodes, 1] output of SpatioTemporalLeakDetector
to pinpoint *which nodes* the GNN thinks are leaking.

Usage
─────
# Localize leaks on a trained experiment, using best_model.pth:
    python localize.py --output_dir outputs/pw5_stable

# Use a specific checkpoint and custom threshold:
    python localize.py --output_dir outputs/pw5_stable \\
                       --checkpoint best_model.pth \\
                       --threshold 0.45 \\
                       --top_k 5

# Save node-score arrays for downstream analysis:
    python localize.py --output_dir outputs/pw5_stable --save_scores

What it produces (saved to <output_dir>/logs/localization/):
  ├── node_leak_heatmap.png        — network graph colored by mean leak score
  ├── top_nodes_bar.png            — ranked bar chart of most suspicious nodes
  ├── scenario_graph_<i>.png       — per-scenario graph snapshots (top 12)
  ├── node_scores_mean.npy         — mean node leak probability [N_nodes]
  └── node_rankings.csv            — node_name, mean_score, rank, above_threshold

Inputs (auto-loaded from output_dir):
  checkpoints/best_model.pth       — trained model weights
  config stored inside checkpoint  — rebuilds PipelineConfig automatically

Architecture reminder
─────────────────────
  SpatioTemporalLeakDetector.forward() → (node_logits [N,1], graph_logits [B])
  node_logits are *raw logits* — apply sigmoid to get leak probabilities.
  The GINEConv layers propagate information across pipes, so a leaking node
  will elevate scores in its neighbours too — that's useful: it identifies
  the leak *zone*, not just the single node.
"""

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from config import PipelineConfig
from dataset import LeakWindowDataset
from models import SpatioTemporalLeakDetector
from topology import TopologyBuilder


# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "axes.edgecolor":   "#30363d",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "legend.frameon":   False,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "#0d1117",
})

CMAP = "plasma"          # colormap: low=purple (safe), high=yellow (leaking)
EDGE_COLOR  = "#30363d"
NODE_RADIUS = 180        # scatter marker size


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeakDB — Node-Level Leak Localization")
    p.add_argument("--output_dir",  type=str, required=True,
                   help="Experiment output dir (same as used in train.py)")
    p.add_argument("--data_root",   type=str, default=None,
                   help="Override config.py data_root path to Parquet files")
    p.add_argument("--inp_file",    type=str, default=None,
                   help="Override config.py inp_file path to EPANET .inp")
    p.add_argument("--checkpoint",  type=str, default="best_model.pth",
                   help="Checkpoint filename inside output_dir/models/")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Sigmoid probability threshold for 'leaking' (default 0.5)")
    p.add_argument("--top_k",       type=int,   default=5,
                   help="How many top-suspect nodes to highlight (default 5)")
    p.add_argument("--max_scenarios", type=int, default=12,
                   help="Max per-scenario graph plots to save (default 12)")
    p.add_argument("--save_scores", action="store_true",
                   help="Also save node_scores_mean.npy for downstream analysis")
    p.add_argument("--split",       type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Which data split to run inference on (default: test)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size",  type=int, default=16)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — graph layout
# ──────────────────────────────────────────────────────────────────────────────

def _circular_layout(n: int) -> np.ndarray:
    """
    Returns (n, 2) positions arranged in a circle.
    Falls back to a grid for larger networks.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return pos


def _hanoi_layout(node_names: List[str]) -> np.ndarray:
    """
    Hanoi network has a known ring-of-rings topology (32 nodes).
    This approximation puts them on a circle with equal spacing.
    Use circular layout as a universal fallback.
    """
    return _circular_layout(len(node_names))


def _get_layout(node_names: List[str]) -> np.ndarray:
    """
    Dispatch to a topology-aware layout if we recognise the network;
    fall back to circular otherwise.
    """
    return _circular_layout(len(node_names))


# ──────────────────────────────────────────────────────────────────────────────
# Inference — collect per-node scores across a full dataset split
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_node_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run forward passes and aggregate per-node leak probabilities.

    Returns
    -------
    all_node_probs  : [n_scenarios, n_nodes]  sigmoid(node_logits)
    all_graph_probs : [n_scenarios]           graph-level leak probability
    all_labels      : [n_scenarios]           ground-truth binary label
    """
    model.eval()
    all_node_probs:  List[np.ndarray] = []
    all_graph_probs: List[float]      = []
    all_labels:      List[float]      = []

    for batch in loader:
        batch = batch.to(device)
        node_logits, graph_logits = model(batch)

        node_probs  = torch.sigmoid(node_logits).squeeze(-1)  # [total_nodes]
        graph_probs = torch.sigmoid(graph_logits)              # [B]

        # Split per-graph using batch.batch membership vector
        batch_vec = batch.batch  # [total_nodes] — which graph each node belongs to
        B = graph_logits.shape[0]

        for g in range(B):
            mask = (batch_vec == g)
            node_p = node_probs[mask].cpu().numpy()            # [N_nodes_in_g]

            # Pad or truncate to n_nodes if topology varies across batches
            if len(node_p) < n_nodes:
                node_p = np.pad(node_p, (0, n_nodes - len(node_p)))
            else:
                node_p = node_p[:n_nodes]

            all_node_probs.append(node_p)
            all_graph_probs.append(graph_probs[g].item())

        all_labels.extend(batch.y.cpu().numpy().tolist())

    return (
        np.array(all_node_probs),   # [S, N]
        np.array(all_graph_probs),  # [S]
        np.array(all_labels),       # [S]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def _draw_network(
    ax: plt.Axes,
    pos: np.ndarray,
    edge_index: np.ndarray,
    node_scores: np.ndarray,
    node_names: List[str],
    threshold: float,
    top_k: int,
    title: str,
    show_labels: bool = True,
) -> None:
    """
    Draw one network graph on `ax`.

    node_scores : [N] float in [0, 1] — probability of leaking
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="#e6edf3", pad=6)

    # Draw edges (only unique undirected pairs)
    seen = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        ax.plot(
            [pos[u, 0], pos[v, 0]],
            [pos[u, 1], pos[v, 1]],
            color=EDGE_COLOR, lw=1.2, zorder=1,
        )

    # Draw nodes colored by score
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap_fn = cm.get_cmap(CMAP)

    colors = [cmap_fn(norm(s)) for s in node_scores]
    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        s=NODE_RADIUS,
        c=node_scores,
        cmap=CMAP,
        vmin=0, vmax=1,
        zorder=3,
        edgecolors="#e6edf3",
        linewidths=0.5,
    )

    # Highlight nodes above threshold with a ring
    top_idx  = np.argsort(node_scores)[::-1][:top_k]
    above_th = np.where(node_scores >= threshold)[0]
    for idx in above_th:
        ax.scatter(
            pos[idx, 0], pos[idx, 1],
            s=NODE_RADIUS * 2.2,
            facecolors="none",
            edgecolors="#ff4444",
            linewidths=2.0,
            zorder=4,
        )

    # Label nodes (suppress for large networks)
    N = len(node_names)
    if show_labels and N <= 40:
        for i, name in enumerate(node_names):
            is_top = i in top_idx
            ax.annotate(
                name,
                xy=(pos[i, 0], pos[i, 1]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7 if N > 20 else 8,
                color="#ff8888" if is_top else "#8b949e",
                fontweight="bold" if is_top else "normal",
            )

    return sc


def plot_heatmap(
    node_names: List[str],
    mean_scores: np.ndarray,
    edge_index: np.ndarray,
    threshold: float,
    top_k: int,
    out_path: str,
) -> None:
    """Mean leak probability over the entire test set — one graph."""
    pos = _get_layout(node_names)

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#0d1117")

    sc = _draw_network(
        ax, pos, edge_index, mean_scores, node_names,
        threshold, top_k,
        title=f"Mean Node Leak Probability  (threshold={threshold:.2f})",
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    cbar.outline.set_edgecolor("#30363d")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
    cbar.set_label("Leak probability", color="#e6edf3")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='none', markeredgecolor='#ff4444',
               markeredgewidth=2, markersize=10,
               label=f'Above threshold ({threshold:.2f})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="#0d1117")
    plt.close(fig)
    print(f"  ✅ Node heatmap        → {out_path}")


def plot_top_nodes(
    node_names: List[str],
    mean_scores: np.ndarray,
    threshold: float,
    top_k: int,
    out_path: str,
) -> None:
    """Horizontal bar chart of nodes ranked by mean leak probability."""
    ranked_idx  = np.argsort(mean_scores)[::-1]
    show_n      = min(len(node_names), max(top_k * 2, 15))
    top_indices = ranked_idx[:show_n]

    names  = [node_names[i] for i in top_indices]
    scores = mean_scores[top_indices]
    colors = ["#ff4444" if s >= threshold else "#1a6fc4" for s in scores]

    fig, ax = plt.subplots(figsize=(9, max(4, show_n * 0.4)))
    fig.patch.set_facecolor("#0d1117")

    bars = ax.barh(range(len(names))[::-1], scores, color=colors, height=0.7)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean leak probability")
    ax.set_title("Node Leak Suspicion Ranking  (test set average)")
    ax.axvline(threshold, color="#ff8800", lw=1.5, ls="--",
               label=f"Threshold = {threshold:.2f}", alpha=0.8)
    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", fontsize=9)

    # Value labels
    for bar, score in zip(bars, scores[::-1]):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", ha="left",
                fontsize=8, color="#e6edf3")

    fig.tight_layout()
    fig.savefig(out_path, facecolor="#0d1117")
    plt.close(fig)
    print(f"  ✅ Node ranking bar    → {out_path}")


def plot_scenario_graphs(
    node_names: List[str],
    all_node_probs: np.ndarray,
    all_graph_probs: np.ndarray,
    all_labels: np.ndarray,
    edge_index: np.ndarray,
    threshold: float,
    top_k: int,
    max_scenarios: int,
    out_dir: str,
) -> None:
    """
    Save individual per-scenario graphs (sorted by graph leak probability,
    highest first). Useful for inspecting the model's spatial reasoning.
    """
    pos = _get_layout(node_names)

    # Prioritise true-positive scenarios (actual leaks caught by model)
    tp_mask = (all_labels == 1) & (all_graph_probs >= threshold)
    tp_idx  = np.where(tp_mask)[0]
    tp_idx  = tp_idx[np.argsort(all_graph_probs[tp_idx])[::-1]]

    fn_mask = (all_labels == 1) & (all_graph_probs < threshold)
    fn_idx  = np.where(fn_mask)[0]
    fn_idx  = fn_idx[np.argsort(all_graph_probs[fn_idx])[::-1]]

    # Mix of TPs and FNs up to max_scenarios
    selected = list(tp_idx[:max_scenarios // 2]) + list(fn_idx[:max_scenarios // 2])
    selected = selected[:max_scenarios]

    cols = min(3, len(selected))
    rows = (len(selected) + cols - 1) // cols

    if len(selected) == 0:
        print("  ⚠️  No scenarios available for scenario grid plot.")
        return

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5, rows * 4.5),
                             squeeze=False)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Per-Scenario Leak Localization\n"
                 "(red ring = node above threshold | sorted by graph leak prob)",
                 color="#e6edf3", fontsize=12, y=1.01)

    for ax_flat_idx, sc_idx in enumerate(selected):
        row, col = divmod(ax_flat_idx, cols)
        ax = axes[row][col]
        node_scores = all_node_probs[sc_idx]
        graph_prob  = all_graph_probs[sc_idx]
        true_label  = int(all_labels[sc_idx])
        detected    = graph_prob >= threshold

        label_str = (
            f"{'✅ TP' if detected and true_label else '🔴 FP' if detected else '❌ FN' if true_label else '✅ TN'}"
            f"  |  Graph prob: {graph_prob:.2f}"
        )
        _draw_network(
            ax, pos, edge_index, node_scores, node_names,
            threshold, top_k,
            title=label_str,
            show_labels=(len(node_names) <= 32),
        )

    # Hide unused axes
    for ax_flat_idx in range(len(selected), rows * cols):
        row, col = divmod(ax_flat_idx, cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "scenario_graph_grid.png")
    fig.savefig(out_path, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Scenario grid       → {out_path}")


def save_node_rankings_csv(
    node_names: List[str],
    mean_scores: np.ndarray,
    threshold: float,
    out_path: str,
) -> None:
    """Write a CSV table: node_name, mean_score, rank, above_threshold."""
    ranked_idx = np.argsort(mean_scores)[::-1]
    import csv
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "node_name", "mean_leak_prob", "above_threshold"])
        for rank, idx in enumerate(ranked_idx, start=1):
            writer.writerow([
                rank,
                node_names[idx],
                f"{mean_scores[idx]:.6f}",
                str(mean_scores[idx] >= threshold),
            ])
    print(f"  ✅ Node rankings CSV   → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    ckpt_path  = os.path.join(output_dir, "checkpoints", args.checkpoint)
    loc_dir    = os.path.join(output_dir, "logs", "localization")
    os.makedirs(loc_dir, exist_ok=True)

    print(f"\n🔍 Leak Localization — {output_dir}")
    print(f"   Checkpoint : {ckpt_path}")
    print(f"   Threshold  : {args.threshold}")
    print(f"   Top-k      : {args.top_k}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device     : {device}\n")

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = PipelineConfig()
    if args.data_root:
        cfg.data_root   = os.path.abspath(os.path.expanduser(args.data_root))
    if args.inp_file:
        cfg.inp_file    = os.path.abspath(os.path.expanduser(args.inp_file))
    if args.batch_size:
        cfg.batch_size  = args.batch_size
    if args.num_workers:
        cfg.num_workers = args.num_workers

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("📂 Building dataset …")
    full_dataset = LeakWindowDataset(cfg)
    total = len(full_dataset)

    # Reproduce the same 70/15/15 split used in train.py
    n_train = int(0.70 * total)
    n_val   = int(0.15 * total)
    n_test  = total - n_train - n_val
    gen     = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=gen
    )
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    dataset   = split_map[args.split]

    print(f"   {args.split} split: {len(dataset):,} windows")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Load topology for first window to get node names ──────────────────────
    sample = full_dataset[0]
    n_nodes = sample.x.shape[0]
    node_names: List[str] = []

    # Try to get node names from the topology builder inside the dataset
    if hasattr(full_dataset, '_topo_map') and full_dataset._topo_map:
        first_key  = next(iter(full_dataset._topo_map))
        node_names, edge_index_t, _pipe_names = full_dataset._topo_map[first_key]
        edge_index = edge_index_t.numpy()
    else:
        # Fallback — generic node names + ring topology
        node_names = [f"J{i+1}" for i in range(n_nodes)]
        edge_index = sample.edge_index.numpy()

    print(f"   Nodes: {n_nodes}  ({', '.join(node_names[:5])}{'…' if n_nodes > 5 else ''})")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n🤖 Loading model from {os.path.basename(ckpt_path)} …")
    model = SpatioTemporalLeakDetector(cfg).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train first with train.py, or point --checkpoint to an existing .pth"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Checkpoints may be raw state_dicts or wrapped dicts
    state = ckpt.get("model_state", ckpt)
    # DDP checkpoints prefix every key with "module." — strip it if present
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print("   Model loaded ✓")

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\n⚡ Running node-level inference on {args.split} split …")
    all_node_probs, all_graph_probs, all_labels = run_node_inference(
        model, loader, device, n_nodes
    )
    print(f"   Collected {all_node_probs.shape[0]:,} scenarios × {n_nodes} nodes")

    # ── Aggregate scores ──────────────────────────────────────────────────────
    # Compute mean separately on leak (label=1) and non-leak (label=0) windows
    leak_mask   = all_labels == 1
    noleak_mask = all_labels == 0

    mean_scores_all   = all_node_probs.mean(axis=0)
    mean_scores_leak  = (all_node_probs[leak_mask].mean(axis=0)
                         if leak_mask.any() else mean_scores_all)
    mean_scores_noleak= (all_node_probs[noleak_mask].mean(axis=0)
                         if noleak_mask.any() else np.zeros(n_nodes))

    top_idx = np.argsort(mean_scores_leak)[::-1][:args.top_k]
    print(f"\n🏆 Top-{args.top_k} most suspicious nodes (leak windows only):")
    for rank, idx in enumerate(top_idx, 1):
        flag = "🔴" if mean_scores_leak[idx] >= args.threshold else "🟡"
        print(f"   {flag} #{rank:2d}  {node_names[idx]:<12s}  "
              f"mean prob = {mean_scores_leak[idx]:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n📊 Generating localization plots …")

    # 1. Network heatmap (mean over leak windows)
    plot_heatmap(
        node_names, mean_scores_leak, edge_index,
        args.threshold, args.top_k,
        os.path.join(loc_dir, "node_leak_heatmap.png"),
    )

    # 2. Contrast heatmap: leak vs no-leak mean scores
    _plot_contrast_heatmap(
        node_names, mean_scores_leak, mean_scores_noleak,
        edge_index, args.threshold, args.top_k,
        os.path.join(loc_dir, "node_leak_contrast.png"),
    )

    # 3. Bar chart
    plot_top_nodes(
        node_names, mean_scores_leak, args.threshold, args.top_k,
        os.path.join(loc_dir, "top_nodes_bar.png"),
    )

    # 4. Scenario-level grid
    plot_scenario_graphs(
        node_names, all_node_probs, all_graph_probs, all_labels,
        edge_index, args.threshold, args.top_k, args.max_scenarios, loc_dir,
    )

    # ── Optional score export ──────────────────────────────────────────────────
    if args.save_scores:
        np.save(os.path.join(loc_dir, "node_scores_all.npy"),    all_node_probs)
        np.save(os.path.join(loc_dir, "node_scores_mean.npy"),   mean_scores_all)
        np.save(os.path.join(loc_dir, "node_scores_leak.npy"),   mean_scores_leak)
        np.save(os.path.join(loc_dir, "graph_probs.npy"),        all_graph_probs)
        np.save(os.path.join(loc_dir, "labels.npy"),             all_labels)
        print(f"  ✅ Score arrays        → {loc_dir}/*.npy")

    save_node_rankings_csv(
        node_names, mean_scores_leak, args.threshold,
        os.path.join(loc_dir, "node_rankings.csv"),
    )

    print(f"\n🎉 Localization complete.  All outputs → {loc_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Contrast heatmap: side-by-side leak vs no-leak
# ──────────────────────────────────────────────────────────────────────────────

def _plot_contrast_heatmap(
    node_names: List[str],
    scores_leak: np.ndarray,
    scores_noleak: np.ndarray,
    edge_index: np.ndarray,
    threshold: float,
    top_k: int,
    out_path: str,
) -> None:
    pos = _get_layout(node_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Node Leak Probability: Leak vs No-Leak Windows",
                 color="#e6edf3", fontsize=13, fontweight="bold")

    titles = [
        f"Leak windows\n(mean prob, threshold={threshold})",
        "No-leak windows\n(mean prob — ideally near 0)",
        "Difference\n(leak − no-leak)",
    ]
    score_sets = [
        scores_leak,
        scores_noleak,
        scores_leak - scores_noleak,
    ]
    vmins = [0, 0, -1]
    vmaxs = [1, 1,  1]
    cmaps = [CMAP, CMAP, "RdBu_r"]

    for ax, title, scores, vmin, vmax, cmap_name in zip(
        axes, titles, score_sets, vmins, vmaxs, cmaps
    ):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color="#e6edf3", fontsize=10, pad=4)
        ax.set_facecolor("#0d1117")

        # Edges
        seen = set()
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                    color=EDGE_COLOR, lw=1.0, zorder=1)

        sc = ax.scatter(
            pos[:, 0], pos[:, 1],
            s=NODE_RADIUS,
            c=scores,
            cmap=cmap_name,
            vmin=vmin, vmax=vmax,
            zorder=3,
            edgecolors="#e6edf3",
            linewidths=0.4,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        cbar.outline.set_edgecolor("#30363d")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")

        # Label top nodes on the leak panel only
        if "Leak windows" in title and len(node_names) <= 40:
            top_idx = np.argsort(scores_leak)[::-1][:top_k]
            for idx in top_idx:
                ax.annotate(
                    node_names[idx],
                    xy=(pos[idx, 0], pos[idx, 1]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7, color="#ff8888", fontweight="bold",
                )

    fig.tight_layout()
    fig.savefig(out_path, facecolor="#0d1117")
    plt.close(fig)
    print(f"  ✅ Contrast heatmap    → {out_path}")


if __name__ == "__main__":
    main()
