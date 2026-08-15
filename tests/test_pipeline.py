"""Tests for the anomaly detection pipeline: feature detection, preprocessing,
model factory (including the LOF novelty gotcha), metrics, and end-to-end train+score."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.config import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from anomaly_detection.data.loader import detect_numeric_features
from anomaly_detection.data.preprocessing import build_preprocessor
from anomaly_detection.engine import train
from anomaly_detection.inference import score
from anomaly_detection.metrics import evaluate_scores
from anomaly_detection.models.factory import build_model


def _make_synthetic_df(n_normal=200, n_anomalies=15, seed=42):
    rng = np.random.default_rng(seed)
    n_features = 4
    A = rng.normal(size=(n_features, n_features)) * 0.8
    cov = A @ A.T + np.eye(n_features) * 0.4
    mean = rng.normal(scale=2, size=n_features)

    normal = rng.multivariate_normal(mean, cov, size=n_normal)
    anomalies = rng.uniform(low=mean - 15, high=mean + 15, size=(n_anomalies, n_features))

    X = np.vstack([normal, anomalies])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df.insert(0, "id", range(len(df)))
    df["label"] = y.astype(int)
    return df


class TestFeatureDetection:
    def test_auto_detects_numeric_columns(self):
        df = _make_synthetic_df()
        cfg = DataConfig(train_csv="x", id_column="id", label_column="label")
        features = detect_numeric_features(df, cfg)
        assert set(features) == {"f0", "f1", "f2", "f3"}

    def test_respects_explicit_override(self):
        df = _make_synthetic_df()
        cfg = DataConfig(train_csv="x", numeric_features=["f0", "f1"])
        features = detect_numeric_features(df, cfg)
        assert features == ["f0", "f1"]

    def test_raises_for_missing_configured_feature(self):
        df = _make_synthetic_df()
        cfg = DataConfig(train_csv="x", numeric_features=["not_a_column"])
        with pytest.raises(ValueError):
            detect_numeric_features(df, cfg)


class TestPreprocessing:
    def test_handles_missing_values(self):
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [5.0, np.nan]])
        preprocessor = build_preprocessor(PreprocessingConfig())
        transformed = preprocessor.fit_transform(X)
        assert not np.isnan(transformed).any()

    def test_scaling_produces_roughly_standardized_output(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=100, scale=20, size=(500, 3))
        preprocessor = build_preprocessor(PreprocessingConfig(scale=True))
        transformed = preprocessor.fit_transform(X)
        assert abs(transformed.mean()) < 0.1
        assert abs(transformed.std() - 1.0) < 0.1

    def test_scale_false_skips_standardization(self):
        X = np.array([[100.0, 200.0], [110.0, 210.0]])
        preprocessor = build_preprocessor(PreprocessingConfig(scale=False))
        transformed = preprocessor.fit_transform(X)
        assert transformed.max() > 10  # unscaled, still in original range


class TestModelFactory:
    def test_isolation_forest_supports_predict_on_new_data(self):
        model = build_model(ModelConfig(type="isolation_forest", contamination=0.1))
        X_train = np.random.default_rng(0).normal(size=(100, 3))
        model.fit(X_train)
        X_new = np.random.default_rng(1).normal(size=(10, 3))
        preds = model.predict(X_new)  # should not raise
        assert set(preds.tolist()) <= {-1, 1}

    def test_lof_is_built_with_novelty_true_so_it_can_score_new_data(self):
        """The exact gotcha this module's docstring warns about: LOF with the
        sklearn default (novelty=False) cannot predict on new data at all."""
        model = build_model(ModelConfig(type="local_outlier_factor", contamination=0.1))
        assert model.novelty is True

        X_train = np.random.default_rng(0).normal(size=(100, 3))
        model.fit(X_train)
        X_new = np.random.default_rng(1).normal(size=(10, 3))
        preds = model.predict(X_new)  # would raise if novelty were False
        assert set(preds.tolist()) <= {-1, 1}

    def test_one_class_svm_maps_contamination_to_nu(self):
        model = build_model(ModelConfig(type="one_class_svm", contamination=0.07))
        assert model.nu == 0.07

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError):
            build_model(ModelConfig(type="not_a_real_model"))


class TestMetrics:
    def test_perfect_separation_gives_auc_one(self):
        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95])
        metrics = evaluate_scores(y_true, scores)
        assert metrics["roc_auc"] == 1.0


class TestEndToEnd:
    @pytest.fixture
    def config(self, tmp_path):
        df = _make_synthetic_df()
        train_df = df[df["label"] == 0].drop(columns=["label"])  # unsupervised: fit on normal data only
        score_df = df.drop(columns=["label"])

        train_csv = tmp_path / "train.csv"
        score_csv = tmp_path / "score.csv"
        train_df.to_csv(train_csv, index=False)
        score_df.to_csv(score_csv, index=False)

        checkpoint_dir = tmp_path / "checkpoints"
        return PipelineConfig(
            data=DataConfig(
                train_csv=str(train_csv), score_csv=str(score_csv),
                id_column="id", random_state=42,
            ),
            preprocessing=PreprocessingConfig(),
            model=ModelConfig(type="isolation_forest", contamination=0.1, params={"n_estimators": 50, "random_state": 42}),
            output=OutputConfig(
                model_dir=str(checkpoint_dir),
                metrics_path=str(checkpoint_dir / "metrics.json"),
                scored_output_path=str(checkpoint_dir / "scored.csv"),
            ),
        )

    def test_train_produces_metrics_and_checkpoint(self, config):
        metrics = train(config)
        assert 0.0 <= metrics["predicted_anomaly_rate"] <= 1.0

        import os
        assert os.path.exists(f"{config.output.model_dir}/{config.output.model_name}")

    def test_score_flags_a_reasonable_fraction_as_anomalies(self, config):
        train(config)
        result = score(config)

        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns
        assert "id" in result.columns
        assert 0 < result["is_anomaly"].mean() < 1  # neither "flags nothing" nor "flags everything"

    def test_scoring_with_missing_feature_column_raises(self, config, tmp_path):
        train(config)

        bad_score_csv = tmp_path / "bad_score.csv"
        pd.DataFrame({"id": [1, 2], "only_one_feature": [1.0, 2.0]}).to_csv(bad_score_csv, index=False)
        config.data.score_csv = str(bad_score_csv)

        with pytest.raises(ValueError):
            score(config)

    def test_full_pipeline_with_labels_reports_evaluation_metrics(self, tmp_path):
        df = _make_synthetic_df()
        train_csv = tmp_path / "train_labeled.csv"
        df.to_csv(train_csv, index=False)

        checkpoint_dir = tmp_path / "checkpoints2"
        config = PipelineConfig(
            data=DataConfig(train_csv=str(train_csv), id_column="id", label_column="label", random_state=42),
            preprocessing=PreprocessingConfig(),
            model=ModelConfig(type="isolation_forest", contamination=0.1, params={"n_estimators": 50, "random_state": 42}),
            output=OutputConfig(
                model_dir=str(checkpoint_dir),
                metrics_path=str(checkpoint_dir / "metrics.json"),
            ),
        )

        metrics = train(config)
        assert "evaluation" in metrics
        assert 0.0 <= metrics["evaluation"]["roc_auc"] <= 1.0
