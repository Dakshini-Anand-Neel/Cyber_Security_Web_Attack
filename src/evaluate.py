"""
src/evaluate.py
===============
Model Evaluation utilities: metrics, confusion matrix, ROC curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from src.paths import REPORT_DIR
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "#0d0221"        # deep dark background
ACCENT  = "#7c3aed"        # violet accent
COLORS  = ["#7c3aed", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#f97316", "#8b5cf6"]


# ══════════════════════════════════════════════════════════════════════════════
# Core metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    average:   str = "weighted",
    zero_div:  int = 0,
) -> Dict[str, float]:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=zero_div),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=zero_div),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=zero_div),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def _set_dark_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0d0221",
        "axes.facecolor":   "#0d0221",
        "axes.edgecolor":   "#4b5563",
        "grid.color":       "#1f2937",
        "text.color":       "#f3f4f6",
        "axes.labelcolor":  "#f3f4f6",
        "xtick.color":      "#9ca3af",
        "ytick.color":      "#9ca3af",
        "font.family":      "DejaVu Sans",
    })


def plot_confusion_matrix(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    class_names:  List[str],
    title:        str = "Confusion Matrix",
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """Plot a styled confusion matrix heatmap."""
    _set_dark_style()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names, yticklabels=class_names,
        cmap="RdPu", linewidths=0.5, linecolor="#1f2937",
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14, color="#c4b5fd")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart comparing models across metrics."""
    _set_dark_style()
    metrics = ["accuracy", "precision", "recall", "f1"]
    models  = results_df.index.tolist()
    x       = np.arange(len(models))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.8), 6))
    for i, metric in enumerate(metrics):
        bars = ax.bar(
            x + i * width, results_df[metric], width,
            label=metric.capitalize(), color=COLORS[i], alpha=0.9, zorder=3,
        )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Evaluation Metrics", fontsize=15,
                 fontweight="bold", color="#c4b5fd")
    ax.legend(loc="lower right", framealpha=0.3)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(
    model:       Any,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: List[str],
    save_path:   Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot one-vs-rest ROC curves (multi-class only)."""
    if not hasattr(model, "predict_proba"):
        print("  Model does not support predict_proba — ROC skipped.")
        return None

    _set_dark_style()
    y_prob = model.predict_proba(X_test)
    n_classes = len(class_names)
    from sklearn.preprocessing import label_binarize
    classes = list(range(n_classes))
    y_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cname}  (AUC={auc:.3f})", color=COLORS[i % len(COLORS)])

    ax.plot([0, 1], [0, 1], "w--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold", color="#c4b5fd")
    ax.legend(loc="lower right", framealpha=0.3)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_text_report(
    best_name:   str,
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    cm:          np.ndarray,
    target_names: List[str],
) -> None:
    """Write classification report + confusion matrix to text file."""
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    path   = REPORT_DIR / "classification_report.txt"
    with open(path, "w") as fh:
        fh.write(f"Best Model: {best_name}\n\n")
        fh.write("Confusion Matrix:\n")
        fh.write(np.array2string(cm) + "\n\n")
        fh.write("Classification Report:\n")
        fh.write(report)
    print(f"  Saved evaluation report → {path}")
