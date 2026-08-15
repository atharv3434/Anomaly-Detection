"""Configuration loading for the anomaly detection pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    train_csv: str
    score_csv: str | None = None
    label_column: str | None = None
    id_column: str | None = None
    numeric_features: list[str] = field(default_factory=list)
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    imputer_strategy: str = "median"
    scale: bool = True


@dataclass
class ModelConfig:
    type: str = "isolation_forest"
    contamination: float = 0.05
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    model_dir: str = "checkpoints"
    model_name: str = "model.joblib"
    metrics_path: str = "checkpoints/metrics.json"
    scored_output_path: str = "checkpoints/scored.csv"


@dataclass
class PipelineConfig:
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    output: OutputConfig
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw.get("data", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            model=ModelConfig(**raw.get("model", {})),
            output=OutputConfig(**raw.get("output", {})),
            log_level=raw.get("logging", {}).get("level", "INFO"),
        )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
