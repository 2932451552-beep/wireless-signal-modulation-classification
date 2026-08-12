"""Validated baseline configuration and safe experiment-result persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BaselineExperimentConfig:
    """Frozen parameters for the first full RadioML baseline."""

    seed: int = 20260812
    epochs: int = 20
    patience: int = 4
    min_delta: float = 0.0001
    learning_rate: float = 0.001
    train_batch_size: int = 256
    evaluation_batch_size: int = 512
    num_workers: int = 0

    def __post_init__(self) -> None:
        integer_values = {
            "seed": self.seed,
            "epochs": self.epochs,
            "patience": self.patience,
            "train_batch_size": self.train_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
        }
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name, value in integer_values.items():
            if name != "seed" and value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and greater than zero")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative")


def prepare_new_run_directory(path: str | Path) -> Path:
    """Create an output directory while refusing to overwrite an existing run."""

    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"experiment directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write UTF-8 JSON through a sibling temporary file and atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=True,
    )
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
