"""Training: fit preprocessing + model, optionally evaluate against labels, persist the bundle."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from anomaly_detection.config import PipelineConfig
from anomaly_detection.data.loader import detect_numeric_features, load_csv
from anomaly_detection.data.preprocessing import build_preprocessor
from anomaly_detection.metrics import evaluate_scores
from anomaly_detection.models.factory import build_model

logger = logging.getLogger(__name__)


def _anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (flips sklearn's decision_function convention,
    where higher = more normal, to match this pipeline's own convention)."""
    return -model.decision_function(X)


def train(config: PipelineConfig) -> dict[str, Any]:
    df = load_csv(config.data.train_csv)
    feature_columns = detect_numeric_features(df, config.data)

    X = df[feature_columns].to_numpy(dtype=float)

    preprocessor = build_preprocessor(config.preprocessing)
    X_processed = preprocessor.fit_transform(X)

    model = build_model(config.model)
    model.fit(X_processed)

    scores = _anomaly_scores(model, X_processed)
    predicted_labels = (model.predict(X_processed) == -1).astype(int)

    metrics: dict[str, Any] = {
        "n_samples": len(df),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "model_type": config.model.type,
        "contamination": config.model.contamination,
        "predicted_anomaly_rate": float(predicted_labels.mean()),
        "score_stats": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        },
    }

    if config.data.label_column and config.data.label_column in df.columns:
        y_true = df[config.data.label_column].to_numpy()
        eval_metrics = evaluate_scores(y_true, scores)
        metrics["evaluation"] = eval_metrics
        logger.info(f"Evaluation against ground-truth labels: {eval_metrics}")
    else:
        logger.info(
            "No label_column configured — training is fully unsupervised, "
            "and no evaluation metrics will be computed (this is expected "
            "for typical anomaly detection use cases)."
        )

    logger.info(f"Trained {config.model.type}: {metrics['predicted_anomaly_rate']:.2%} flagged as anomalies")

    _persist(preprocessor, model, feature_columns, metrics, config)
    return metrics


def _persist(preprocessor, model, feature_columns: list[str], metrics: dict[str, Any], config: PipelineConfig) -> None:
    model_dir = Path(config.output.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = model_dir / config.output.model_name
    joblib.dump({
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": feature_columns,
        "model_type": config.model.type,
    }, bundle_path)
    logger.info(f"Saved model bundle to {bundle_path}")

    metrics_path = Path(config.output.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
