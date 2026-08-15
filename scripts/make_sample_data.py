"""Generate synthetic tabular data with real, correlated normal structure and
injected anomalies (both marginal outliers and correlation-breaking ones) —
enough real signal for the pipeline to demonstrate genuine detection, not just
run without crashing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]

N_FEATURES = 6


def _make_normal_data(n, mean, cov, rng):
    return rng.multivariate_normal(mean=mean, cov=cov, size=n)


def generate(n_train_normal=1500, n_score_normal=300, n_score_anomalies=20):
    rng = np.random.default_rng(42)

    A = rng.normal(size=(N_FEATURES, N_FEATURES)) * 0.8
    cov = A @ A.T + np.eye(N_FEATURES) * 0.4
    mean = rng.normal(scale=2, size=N_FEATURES)
    marginal_stds = np.sqrt(np.diag(cov))

    columns = [f"feature_{i}" for i in range(N_FEATURES)]

    # Training data: normal only (mimics a real deployment where you mostly
    # have "business as usual" data to fit on, with few or no known anomalies).
    train_data = _make_normal_data(n_train_normal, mean, cov, rng)
    train_df = pd.DataFrame(train_data, columns=columns)
    train_df.insert(0, "id", [f"train_{i}" for i in range(len(train_df))])
    train_df.to_csv(ROOT / "data" / "raw" / "train.csv", index=False)

    # Score data: a mix of normal points and two kinds of anomalies, with
    # ground-truth labels included so you can verify detection quality
    # yourself (labels aren't required for scoring — this is just for the demo).
    normal_score = _make_normal_data(n_score_normal, mean, cov, rng)

    marginal_anomalies = []
    for _ in range(n_score_anomalies // 2):
        point = rng.multivariate_normal(mean, cov)
        idx = rng.choice(N_FEATURES, size=rng.integers(1, 3), replace=False)
        point[idx] += rng.choice([-1, 1], size=len(idx)) * rng.uniform(5, 9, size=len(idx))
        marginal_anomalies.append(point)

    correlation_anomalies = [
        rng.normal(loc=mean, scale=marginal_stds) for _ in range(n_score_anomalies - len(marginal_anomalies))
    ]

    score_data = np.vstack([normal_score, marginal_anomalies, correlation_anomalies])
    score_labels = np.concatenate([
        np.zeros(len(normal_score)),
        np.ones(len(marginal_anomalies)),
        np.ones(len(correlation_anomalies)),
    ])

    score_df = pd.DataFrame(score_data, columns=columns)
    score_df.insert(0, "id", [f"score_{i}" for i in range(len(score_df))])
    score_df["true_label"] = score_labels.astype(int)  # for your own reference; not read by the pipeline

    idx = rng.permutation(len(score_df))
    score_df = score_df.iloc[idx].reset_index(drop=True)
    score_df.to_csv(ROOT / "data" / "raw" / "score.csv", index=False)

    print(f"Wrote {len(train_df)} training rows (all normal) to data/raw/train.csv")
    print(f"Wrote {len(score_df)} scoring rows ({int(score_labels.sum())} true anomalies) to data/raw/score.csv")


if __name__ == "__main__":
    generate()
