"""Evaluation metrics — only usable when ground-truth labels are available
(optional; unsupervised anomaly detection doesn't require them to run)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_scores(y_true: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, float]:
    """@param anomaly_scores higher = more anomalous."""
    return {
        "roc_auc": float(roc_auc_score(y_true, anomaly_scores)),
        "average_precision": float(average_precision_score(y_true, anomaly_scores)),
    }
