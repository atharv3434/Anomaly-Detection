# Anomaly Detection Pipeline

A production-ready anomaly detection pipeline: preprocessing, a swappable
model registry (Isolation Forest / Local Outlier Factor / One-Class SVM),
model persistence, and a scoring CLI for new incoming data — with optional
evaluation when ground-truth labels happen to be available.

> Looking for the conceptual walkthrough comparing these methods (and
> classical statistical ones) with visualizations? See this workspace's
> separate `anomaly-detection-tutorial` project. This one is the
> production-shaped version: config-driven, tested, and built to run
> unsupervised against real incoming data, not to teach the concepts.

## Features

- **One config, three algorithms** — `isolation_forest`, `local_outlier_factor`,
  or `one_class_svm`, swappable via `model.type`, with `contamination` mapped
  correctly to whatever parameter each algorithm actually uses internally
- **Handles a genuine LOF gotcha correctly**: scikit-learn's
  `LocalOutlierFactor` defaults to `novelty=False`, which can *only* score the
  exact data it was fit on — calling `.predict()` on new data raises an error.
  This pipeline always builds it with `novelty=True` so it can actually score
  incoming data in production, and has a test that would fail if that
  regressed
- **Missing-value imputation + standardization**, fit once and persisted
  alongside the model so scoring uses the exact same transformation as training
- **Unsupervised by default** — trains without needing any labels, the normal
  case for real anomaly detection — with optional evaluation (ROC-AUC,
  average precision) if you do have a labeled validation set
- **CLI**: `anomaly-detection train` to fit and persist, `anomaly-detection
  score` to run the trained model against new data
- **Tests**: 15 tests covering feature detection, preprocessing, all three
  model types (including a dedicated test for the LOF novelty gotcha), and
  full end-to-end train→score runs

## Project Structure

```
anomaly-detection-pipeline/
├── configs/default.yaml
├── data/raw/                    # train.csv, score.csv
├── checkpoints/                 # persisted model bundle + metrics + scored output
├── scripts/make_sample_data.py  # synthetic data with real correlated structure
├── src/anomaly_detection/
│   ├── config.py                 # config dataclasses + YAML loading
│   ├── cli.py                     # `anomaly-detection` command line entrypoint
│   ├── engine.py                  # training: fit preprocessing + model, persist
│   ├── inference.py               # scoring: load bundle, score new data
│   ├── metrics.py                 # optional evaluation against labels
│   ├── data/
│   │   ├── loader.py              # CSV loading, feature auto-detection
│   │   └── preprocessing.py       # imputation + scaling pipeline
│   └── models/factory.py         # model registry with per-algorithm quirks handled
├── tests/
├── Dockerfile
└── pyproject.toml
```

## Quick Start

```bash
# 1. Install (editable, with dev deps)
pip install -e ".[dev]"

# 2. Generate synthetic data: mostly-normal training data, plus a scoring set
#    with two kinds of injected anomalies (obvious outliers and subtler
#    correlation-breaking ones)
python scripts/make_sample_data.py

# 3. Train (unsupervised — fits on data/raw/train.csv, no labels needed)
anomaly-detection train --config configs/default.yaml

# 4. Score new data
anomaly-detection score --config configs/default.yaml
```

## Using Your Own Data

```
data/raw/
├── train.csv    # mostly-normal data to fit the detector on
└── score.csv     # new data to score
```

Update `configs/default.yaml`:

```yaml
data:
  id_column: "id"              # optional; carried through to scored output
  label_column: null            # set this only if train.csv has ground-truth labels for evaluation
  numeric_features: []          # empty = auto-detect all numeric columns
```

Feature columns are auto-detected from dtype, or specify them explicitly via
`data.numeric_features` for more control.

## Choosing a Model

```yaml
model:
  type: "isolation_forest"      # isolation_forest | local_outlier_factor | one_class_svm
  contamination: 0.05            # expected fraction of anomalies
```

- **`isolation_forest`** — good general-purpose default; fast, handles higher
  dimensions well, particularly good at catching individually-extreme values
- **`local_outlier_factor`** — better than Isolation Forest at catching
  anomalies that are only unusual in combination with other features
  (correlation-breaking anomalies), or where "normal" density varies across
  the data; more compute per prediction
- **`one_class_svm`** — a smooth decision boundary approach; works well on
  smaller/medium datasets, more sensitive to hyperparameter choices

If you have any labeled anomalies at all, set `data.label_column` and compare
`roc_auc`/`average_precision` across model types on your own data before
picking one — the "best" algorithm genuinely depends on what kind of
anomalies matter for your use case, not on which one is fastest or most popular.

## Testing

```bash
pytest tests/ -v --cov=anomaly_detection
```

## Running with Docker

```bash
docker build -t anomaly-detection .
docker run -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints anomaly-detection train
docker run -v $(pwd)/checkpoints:/app/checkpoints anomaly-detection score
```


