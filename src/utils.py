"""Utility helpers for seeding and configuration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_result_dirs() -> None:
    """Create result output directories if they do not exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def linear_epsilon(
    episode: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_min: float,
) -> float:
    """Linear epsilon decay over training episodes."""
    if total_episodes <= 1:
        return epsilon_min
    progress = min(episode / (total_episodes - 1), 1.0)
    return epsilon_start + progress * (epsilon_min - epsilon_start)
