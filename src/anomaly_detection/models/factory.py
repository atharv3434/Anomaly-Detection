"""Model factory for anomaly detection algorithms.

All three supported models expose the same scikit-learn outlier-detection
API surface (`decision_function` → higher means more normal; `predict` →
1 for inlier, -1 for outlier), so the rest of the pipeline can treat them
uniformly. Getting there requires model-specific care:

- IsolationForest: works out of the box for scoring new data.
- LocalOutlierFactor: by default (`novelty=False`) it can ONLY score the
  exact data it was fit on — calling `.predict()` on new data raises an
  error. For a production pipeline that needs to score future incoming
  data, `novelty=True` is required. This is a genuinely easy mistake to
  make (LOF's default is fine for one-off analysis, wrong for a serving
  pipeline) and is called out explicitly here.
- OneClassSVM: uses `nu` instead of `contamination` for the same concept
  (expected fraction of outliers) — mapped automatically here so the
  config only needs one consistent knob.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from anomaly_detection.config import ModelConfig


def build_model(config: ModelConfig) -> BaseEstimator:
    params = dict(config.params)

    if config.type == "isolation_forest":
        params.setdefault("contamination", config.contamination)
        return IsolationForest(**params)

    if config.type == "local_outlier_factor":
        params.setdefault("contamination", config.contamination)
        params["novelty"] = True  # required to score data other than what it was fit on
        return LocalOutlierFactor(**params)

    if config.type == "one_class_svm":
        params.setdefault("nu", config.contamination)
        params.pop("contamination", None)  # OneClassSVM doesn't accept this param name
        return OneClassSVM(**params)

    raise ValueError(
        f"Unknown model type '{config.type}'. "
        f"Available: isolation_forest, local_outlier_factor, one_class_svm"
    )
