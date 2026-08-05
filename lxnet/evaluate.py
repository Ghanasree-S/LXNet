"""Metrics, cross-fold summaries and the Wilcoxon signed-rank comparison.

Macro averaging throughout: the dataset runs 1340 Normal against 544 Chest
Changes, so micro-averaged numbers flatter any model that neglects small
classes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ALPHA = 0.05


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Accuracy plus macro precision/recall/F1 and a confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    labels = list(range(num_classes)) if num_classes else None
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if class_names is not None:
        metrics["report"] = classification_report(
            y_true, y_pred, target_names=list(class_names), zero_division=0, labels=labels
        )
    return metrics


def wilcoxon_compare(a: Sequence[float], b: Sequence[float], alpha: float = ALPHA) -> dict:
    """Wilcoxon signed-rank test on paired per-fold scores.

    Paired because both models see identical folds, and non-parametric because
    five folds is far too few to justify assuming normality.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must match in length: {a.shape} vs {b.shape}")

    differences = a - b
    if np.allclose(differences, 0):
        # scipy raises on an all-zero difference vector; the answer is "no effect".
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "median_difference": 0.0,
        }

    statistic, p_value = wilcoxon(a, b)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "median_difference": float(np.median(differences)),
    }


def summarise_folds(folds: Sequence[dict]) -> dict:
    """Mean and standard deviation of every scalar metric across folds."""
    if not folds:
        raise ValueError("cannot summarise an empty list of folds")

    summary: dict[str, float] = {}
    for key in folds[0]:
        values = [f[key] for f in folds if key in f]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary
