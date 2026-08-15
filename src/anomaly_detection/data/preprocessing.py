"""Preprocessing: imputation + scaling, as a fitted, persistable sklearn Pipeline."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anomaly_detection.config import PreprocessingConfig


def build_preprocessor(config: PreprocessingConfig) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy=config.imputer_strategy))]
    if config.scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
