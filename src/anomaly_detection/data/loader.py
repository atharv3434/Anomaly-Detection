"""Data loading and feature-type auto-detection."""
from __future__ import annotations

import logging

import pandas as pd

from anomaly_detection.config import DataConfig

logger = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_numeric_features(df: pd.DataFrame, config: DataConfig) -> list[str]:
    """Return the numeric feature columns to use, respecting an explicit
    config override, otherwise auto-detecting numeric dtypes and excluding
    id/label columns."""
    if config.numeric_features:
        missing = [c for c in config.numeric_features if c not in df.columns]
        if missing:
            raise ValueError(f"Configured numeric_features not found in data: {missing}")
        return list(config.numeric_features)

    exclude = set()
    if config.id_column:
        exclude.add(config.id_column)
    if config.label_column:
        exclude.add(config.label_column)

    numeric_features = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_features:
        raise ValueError(
            "No numeric feature columns found — check your data or set "
            "data.numeric_features explicitly in the config."
        )

    logger.info(f"Auto-detected {len(numeric_features)} numeric feature(s): {numeric_features}")
    return numeric_features
