"""Score new (unseen) data with a trained model bundle."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomaly_detection.config import PipelineConfig

logger = logging.getLogger(__name__)


def _anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    return -model.decision_function(X)


def score(config: PipelineConfig) -> pd.DataFrame:
    """Load the trained bundle, score `data.score_csv`, and write results
    (original id column if configured, anomaly score, and boolean flag)."""
    if not config.data.score_csv:
        raise ValueError("config.data.score_csv must be set to run scoring")

    bundle_path = Path(config.output.model_dir) / config.output.model_name
    bundle = joblib.load(bundle_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = pd.read_csv(config.data.score_csv)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Score data is missing feature column(s) used in training: {missing}")

    X = df[feature_columns].to_numpy(dtype=float)
    X_processed = preprocessor.transform(X)

    scores = _anomaly_scores(model, X_processed)
    is_anomaly = (model.predict(X_processed) == -1)

    result = pd.DataFrame({"anomaly_score": scores, "is_anomaly": is_anomaly})
    if config.data.id_column and config.data.id_column in df.columns:
        result.insert(0, config.data.id_column, df[config.data.id_column])

    out_path = Path(config.output.scored_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info(
        f"Scored {len(result)} rows, {int(is_anomaly.sum())} flagged as anomalies "
        f"({is_anomaly.mean():.2%}) — wrote {out_path}"
    )

    return result
