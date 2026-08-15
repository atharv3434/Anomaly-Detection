"""Command-line interface for the anomaly detection pipeline."""

from __future__ import annotations

import json
import logging

import click

from anomaly_detection.config import PipelineConfig, setup_logging
from anomaly_detection.engine import train as train_fn
from anomaly_detection.inference import score as score_fn

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Production anomaly detection pipeline (Isolation Forest / LOF / One-Class SVM)."""


@cli.command()
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
def train(config_path: str) -> None:
    """Fit the preprocessing + model pipeline and persist it."""
    config = PipelineConfig.from_yaml(config_path)
    setup_logging(config.log_level)
    metrics = train_fn(config)
    summary = {
        "predicted_anomaly_rate": metrics["predicted_anomaly_rate"],
        "evaluation": metrics.get("evaluation"),
    }
    click.echo(json.dumps(summary, indent=2))


@cli.command()
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
def score(config_path: str) -> None:
    """Score new data with the trained model."""
    config = PipelineConfig.from_yaml(config_path)
    setup_logging(config.log_level)
    result = score_fn(config)
    click.echo(f"Wrote {len(result)} scored rows to {config.output.scored_output_path}")
    click.echo(result.head().to_string(index=False))


if __name__ == "__main__":
    cli()
