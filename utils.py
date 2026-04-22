"""
utils.py
────────
Shared utilities for training and evaluation.

  EarlyStoppingF1   — Halt when validation F1 stops improving.
  compute_metrics   — F1, Precision, Recall, ROC-AUC from probability arrays.
  save_checkpoint   — Save model + optimizer + epoch to a single file.
  load_checkpoint   — Restore from checkpoint dict.
"""

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStoppingF1:
    """
    Monitors validation F1 score.  Triggers `early_stop = True` when F1 has
    not improved by at least `min_delta` for `patience` consecutive epochs.

    Best model weights are saved to `path` every time a new best is recorded.

    Args:
        patience  : epochs to wait before stopping
        min_delta : minimum improvement to count as a real gain
        path      : file path for saving best weights (state_dict only)
    """

    def __init__(
        self,
        patience:  int   = 15,
        min_delta: float = 0.001,
        path:      str   = "best_model.pth",
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.counter    = 0
        self.best_f1    = -1.0
        self.early_stop = False

    def __call__(self, current_f1: float, model: nn.Module) -> None:
        if current_f1 > self.best_f1 + self.min_delta:
            print(
                f"  📈 Val F1 improved "
                f"{self.best_f1:.4f} → {current_f1:.4f}  |  saving weights…"
            )
            self.best_f1 = current_f1
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f"  ⚠️  No improvement  "
                f"({current_f1:.4f} ≤ {self.best_f1:.4f} + δ)  "
                f"|  patience {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:      np.ndarray,
    y_pred_prob: np.ndarray,
    threshold:   float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true      : float array with ground-truth labels (0.0 / 1.0)
        y_pred_prob : float array with predicted probabilities in [0, 1]
        threshold   : decision threshold for converting probs to labels

    Returns:
        dict with keys: f1, precision, recall, roc_auc
        If y_pred_prob contains NaN (gradient explosion), all metrics return 0.0
        and a warning is printed. This prevents a crash so training can continue.
    """
    # ── NaN guard ──────────────────────────────────────────────────────────────
    # NaN in predictions means the model weights went NaN during training
    # (gradient explosion that AMP scaler failed to catch).  Return zeroed
    # metrics instead of crashing so early stopping can still fire cleanly.
    nan_frac = np.isnan(y_pred_prob).mean()
    if nan_frac > 0:
        print(
            f"\n  ⚠️  NaN detected in {nan_frac*100:.1f}% of predictions. "
            "Gradient explosion suspected. Returning zero metrics for this epoch. "
            "Consider reducing --lr or --grad_clip."
        )
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "roc_auc": 0.0}

    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred_prob > threshold).astype(int)

    # ROC-AUC requires at least one positive example
    has_pos = y_true_bin.sum() > 0
    has_neg = (1 - y_true_bin).sum() > 0

    return {
        "f1":        f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall":    recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_true_bin, y_pred_prob)
            if has_pos and has_neg
            else 0.0
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   Dict[str, float],
) -> None:
    """Save a full training checkpoint (model + optimiser state + metadata)."""
    torch.save(
        {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics":        metrics,
        },
        path,
    )


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device:    torch.device                 = torch.device("cpu"),
) -> Dict:
    """
    Restore model (and optionally optimizer) from a checkpoint file.

    Returns the checkpoint dict so the caller can inspect epoch / metrics.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
