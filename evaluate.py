"""
evaluate.py
───────────
Standalone evaluation and visualisation script.  Run after training:

    python evaluate.py --output_dir ~/LeakDB/outputs

What it produces (all saved to <output_dir>/logs/):
  ├── confusion_matrix.png   — heatmap with TP/FP/TN/FN counts and rates
  ├── roc_curve.png          — ROC curve with AUC annotation
  ├── pr_curve.png           — Precision-Recall curve with AP annotation
  ├── threshold_sweep.png    — F1/Precision/Recall vs threshold
  ├── training_history.png   — train loss, val loss, val F1 across epochs
  └── evaluation_report.txt  — full numeric summary

Everything is also printed to stdout so you get results immediately even
if you're running on a headless server.

Inputs (written by train.py):
  logs/test_probs.npy   — model sigmoid outputs for the test set  [N]
  logs/test_labels.npy  — binary ground-truth for the test set     [N]
  logs/training_log.csv — per-epoch training metrics

If you want to re-run inference on a different split, pass --rerun_inference
and the script will rebuild the dataset from the Parquet files and run
the model forward pass again.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    matthews_corrcoef,
)


# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

ACCENT   = "#1a6fc4"     # blue
POSITIVE = "#d44d3a"     # red  — leak
NEGATIVE = "#4d9e6e"     # green — no leak

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linewidth":   0.6,
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "legend.frameon":   False,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeakDB Evaluation & Visualisation")
    p.add_argument(
        "--output_dir", type=str,
        default=os.path.expanduser("~/LeakDB/outputs"),
        help="Same output_dir used in train.py",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for binary predictions (default 0.5)",
    )
    p.add_argument(
        "--rerun_inference", action="store_true",
        help="Re-run model inference instead of loading saved .npy files",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Load predictions
# ──────────────────────────────────────────────────────────────────────────────

def load_predictions(logs_dir: str):
    probs_path  = os.path.join(logs_dir, "test_probs.npy")
    labels_path = os.path.join(logs_dir, "test_labels.npy")

    if not os.path.exists(probs_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Could not find test_probs.npy / test_labels.npy in {logs_dir}.\n"
            "These are saved automatically by train.py.  Re-run training, or\n"
            "pass --rerun_inference to generate them now."
        )
    probs  = np.load(probs_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.float32)
    print(f"  Loaded {len(probs):,} test-set predictions from {logs_dir}")
    return probs, labels


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_all(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    y_bin  = y_true.astype(int)

    cm = confusion_matrix(y_bin, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, int(cm[0, 0]))

    has_pos = y_bin.sum() > 0
    has_neg = (1 - y_bin).sum() > 0

    return {
        "threshold": threshold,
        "accuracy":  accuracy_score(y_bin, y_pred),
        "f1":        f1_score(y_bin, y_pred, zero_division=0),
        "precision": precision_score(y_bin, y_pred, zero_division=0),
        "recall":    recall_score(y_bin, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_bin, y_prob) if has_pos and has_neg else 0.0,
        "avg_prec":  average_precision_score(y_bin, y_prob) if has_pos else 0.0,
        "mcc":       matthews_corrcoef(y_bin, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "npv":         tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "cm": cm,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(m: dict, out_path: str) -> None:
    cm = m["cm"]
    labels_txt = ["No leak (0)", "Leak (1)"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_txt); ax.set_yticklabels(labels_txt)
    ax.set_xlabel("Predicted label");  ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    ax.grid(False)

    totals = cm.sum(axis=1, keepdims=True)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = 100 * count / max(totals[i, 0], 1)
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color=color, fontweight="bold")

    # Annotate quadrant names in corners
    for (xi, yi, name, clr) in [
        (0.02, 0.02, "TN", NEGATIVE),
        (0.98, 0.02, "FP", POSITIVE),
        (0.02, 0.98, "FN", POSITIVE),
        (0.98, 0.98, "TP", NEGATIVE),
    ]:
        ax.text(xi, yi, name, transform=ax.transAxes,
                fontsize=9, color=clr, va="bottom" if yi < 0.5 else "top",
                ha="left"  if xi < 0.5 else "right", alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Confusion matrix   → {out_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   auc: float, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ ROC curve          → {out_path}")


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  avg_prec: float, threshold: float, out_path: str) -> None:
    prec, rec, thresholds = precision_recall_curve(y_true.astype(int), y_prob)
    # Mark operating point
    pred_at_thresh = (y_prob >= threshold).astype(int)
    op_prec = precision_score(y_true.astype(int), pred_at_thresh, zero_division=0)
    op_rec  = recall_score(y_true.astype(int),    pred_at_thresh, zero_division=0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, color=POSITIVE, lw=2,
            label=f"PR curve (AP = {avg_prec:.4f})")
    ax.scatter([op_rec], [op_prec], color=ACCENT, s=80, zorder=5,
               label=f"Threshold = {threshold:.2f}")
    ax.fill_between(rec, prec, alpha=0.08, color=POSITIVE)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ PR curve           → {out_path}")


def plot_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray,
                         threshold: float, out_path: str) -> None:
    """Plot F1, Precision, Recall and Accuracy across decision thresholds."""
    thresholds = np.linspace(0.01, 0.99, 200)
    y_bin = y_true.astype(int)

    f1s, precs, recs, accs = [], [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_bin, yp, zero_division=0))
        precs.append(precision_score(y_bin, yp, zero_division=0))
        recs.append(recall_score(y_bin, yp, zero_division=0))
        accs.append(accuracy_score(y_bin, yp))

    # Best F1 threshold
    best_idx = int(np.argmax(f1s))
    best_t   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, f1s,   label="F1",        color=ACCENT,    lw=2)
    ax.plot(thresholds, precs, label="Precision",  color=POSITIVE,  lw=1.5, ls="--")
    ax.plot(thresholds, recs,  label="Recall",     color=NEGATIVE,  lw=1.5, ls=":")
    ax.plot(thresholds, accs,  label="Accuracy",   color="#888",    lw=1.2, ls="-.")

    ax.axvline(threshold, color="gray",  lw=1,   ls="--", alpha=0.7,
               label=f"Current threshold ({threshold:.2f})")
    ax.axvline(best_t,    color=ACCENT,  lw=1.5, ls="--",
               label=f"Best F1 threshold ({best_t:.2f})")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs. decision threshold")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Threshold sweep    → {out_path}")
    print(f"     Best F1 @ threshold {best_t:.3f}  →  F1 = {f1s[best_idx]:.4f}")

    return best_t, f1s[best_idx]


def plot_training_history(log_csv: str, out_path: str,
                          experiment_name: str = "") -> None:
    """
    3-panel convergence figure saved to logs/:
      Panel 1 — Train loss vs Val loss
      Panel 2 — Val F1 with best-epoch marker + ROC-AUC
      Panel 3 — Val Precision vs Val Recall per epoch
    """
    import csv as csv_mod

    if not os.path.exists(log_csv):
        print(f"  ⚠️  training_log.csv not found — skipping history plot.")
        return

    rows = []
    with open(log_csv, newline="") as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except ValueError:
                continue   # skip NaN rows

    if not rows:
        return

    epochs     = [r["epoch"]       for r in rows]
    train_loss = [r["train_loss"]  for r in rows]
    val_loss   = [r["val_loss"]    for r in rows]
    val_f1     = [r["val_f1"]      for r in rows]
    val_auc    = [r["val_roc_auc"] for r in rows]
    val_prec   = [r["val_precision"] for r in rows]
    val_rec    = [r["val_recall"]    for r in rows]

    best_idx = int(np.argmax(val_f1))
    best_ep  = epochs[best_idx]
    best_f1  = val_f1[best_idx]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    title = f"Training convergence — {experiment_name}" if experiment_name \
            else "Training convergence"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: Loss ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, color=ACCENT,   lw=2,      label="Train loss")
    ax.plot(epochs, val_loss,   color=POSITIVE, lw=2, ls="--", label="Val loss")
    ax.axvline(best_ep, color="gray", lw=1, ls=":", alpha=0.6)
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.set_title("Loss")

    # ── Panel 2: F1 + AUC ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, val_f1,  color=NEGATIVE, lw=2,      label="Val F1")
    ax.plot(epochs, val_auc, color=ACCENT,   lw=1.5, ls="--", label="Val AUC")
    ax.scatter([best_ep], [best_f1], color=NEGATIVE, s=60, zorder=5,
               label=f"Best F1={best_f1:.4f} (ep {int(best_ep)})")
    ax.axvline(best_ep, color="gray", lw=1, ls=":", alpha=0.6)
    ax.axhline(best_f1, color=NEGATIVE, lw=0.8, ls=":", alpha=0.4)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.set_title("Val F1 and ROC-AUC")

    # ── Panel 3: Precision / Recall ────────────────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, val_prec, color="#7F77DD", lw=2,      label="Precision")
    ax.plot(epochs, val_rec,  color="#BA7517", lw=2, ls="--", label="Recall")
    ax.axvline(best_ep, color="gray", lw=1, ls=":", alpha=0.6)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Score")
    ax.set_xlabel("Epoch")
    ax.legend(loc="lower right")
    ax.set_title("Precision vs Recall")

    # Annotate best epoch across all panels
    for ax in axes:
        ax.annotate(
            f"best\nep {int(best_ep)}",
            xy=(best_ep, ax.get_ylim()[0]),
            xytext=(best_ep + max(1, len(epochs) * 0.02), ax.get_ylim()[0]),
            fontsize=8, color="gray", va="bottom",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Training history   → {out_path}")


def plot_all(output_dir: str, threshold: float = 0.5,
             experiment_name: str = "") -> None:
    """
    Entry point called by train.py at the end of training.
    Generates every plot and the evaluation report from whatever
    data exists in logs/.  Safe to call even if test_probs.npy
    doesn't exist yet — it skips those plots gracefully.

    Also callable standalone:
        from evaluate import plot_all
        plot_all("/path/to/outputs/run_A_pw10")
    """
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # ── Training convergence — always available once epoch 1 finishes ─────────
    plot_training_history(
        os.path.join(logs_dir, "training_log.csv"),
        os.path.join(logs_dir, "training_history.png"),
        experiment_name=experiment_name,
    )

    # ── Test-set plots — only if inference has been run ───────────────────────
    probs_path  = os.path.join(logs_dir, "test_probs.npy")
    labels_path = os.path.join(logs_dir, "test_labels.npy")

    if not (os.path.exists(probs_path) and os.path.exists(labels_path)):
        print("  ⚠️  test_probs.npy / test_labels.npy not found — "
              "skipping test-set plots.")
        return

    y_prob = np.load(probs_path).astype(np.float32)
    y_true = np.load(labels_path).astype(np.float32)

    # NaN guard
    if np.isnan(y_prob).any():
        print("  ⚠️  NaN in test predictions — skipping test-set plots.")
        return

    m = compute_all(y_true, y_prob, threshold=threshold)

    plot_confusion_matrix(m, os.path.join(logs_dir, "confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, m["roc_auc"],
                   os.path.join(logs_dir, "roc_curve.png"))
    plot_pr_curve(y_true, y_prob, m["avg_prec"], threshold,
                  os.path.join(logs_dir, "pr_curve.png"))
    best_t, best_f1 = plot_threshold_sweep(
        y_true, y_prob, threshold,
        os.path.join(logs_dir, "threshold_sweep.png"),
    )
    print_and_save_report(
        m, best_t, best_f1,
        os.path.join(logs_dir, "evaluation_report.txt"),
    )

    print(f"\n  All plots → {logs_dir}")



# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def print_and_save_report(m: dict, best_t: float, best_f1_at_best_t: float,
                          out_path: str) -> None:
    total = m["tp"] + m["tn"] + m["fp"] + m["fn"]
    lines = [
        "=" * 60,
        "  EVALUATION REPORT  —  LeakDB Leak Detection",
        "=" * 60,
        f"  Test windows         : {total:,}",
        f"  Decision threshold   : {m['threshold']:.2f}",
        "",
        "  ── Classification metrics ──────────────────────────────",
        f"  Accuracy             : {m['accuracy']:.4f}",
        f"  F1 Score             : {m['f1']:.4f}",
        f"  Precision            : {m['precision']:.4f}",
        f"  Recall (sensitivity) : {m['recall']:.4f}",
        f"  Specificity          : {m['specificity']:.4f}",
        f"  NPV                  : {m['npv']:.4f}",
        f"  MCC                  : {m['mcc']:.4f}",
        "",
        "  ── Ranking metrics ────────────────────────────────────",
        f"  ROC-AUC              : {m['roc_auc']:.4f}",
        f"  Avg Precision (AP)   : {m['avg_prec']:.4f}",
        "",
        "  ── Confusion matrix counts ────────────────────────────",
        f"  TP  {m['tp']:>8,}   FN  {m['fn']:>8,}",
        f"  FP  {m['fp']:>8,}   TN  {m['tn']:>8,}",
        "",
        "  ── Threshold recommendation ───────────────────────────",
        f"  Best F1 threshold    : {best_t:.3f}",
        f"  F1 at that threshold : {best_f1_at_best_t:.4f}",
        "=" * 60,
    ]
    for line in lines:
        print(line)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  Full report → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    logs_dir   = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    print(f"\n📊 Evaluate — loading predictions from {logs_dir}\n")

    y_prob, y_true = load_predictions(logs_dir)
    m = compute_all(y_true, y_prob, threshold=args.threshold)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        m,
        os.path.join(logs_dir, "confusion_matrix.png"),
    )
    plot_roc_curve(
        y_true, y_prob, m["roc_auc"],
        os.path.join(logs_dir, "roc_curve.png"),
    )
    plot_pr_curve(
        y_true, y_prob, m["avg_prec"], args.threshold,
        os.path.join(logs_dir, "pr_curve.png"),
    )
    best_t, best_f1 = plot_threshold_sweep(
        y_true, y_prob, args.threshold,
        os.path.join(logs_dir, "threshold_sweep.png"),
    )
    plot_training_history(
        os.path.join(logs_dir, "training_log.csv"),
        os.path.join(logs_dir, "training_history.png"),
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print_and_save_report(
        m, best_t, best_f1,
        os.path.join(logs_dir, "evaluation_report.txt"),
    )
    print(f"\n🎉 All plots saved to {logs_dir}")


if __name__ == "__main__":
    main()
