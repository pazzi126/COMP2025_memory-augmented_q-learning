"""Metric calculations for experiment analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


CHECKPOINTS = [500, 1000, 2000, 5000, 10000]


def success_rate(series: pd.Series) -> float:
    """Compute mean success rate."""
    return float(series.mean())


def learning_curve_auc(episode_success: pd.Series) -> float:
    """Area under the episode success-rate learning curve."""
    values = episode_success.astype(float).to_numpy()
    if len(values) == 0:
        return 0.0
    return float(np.trapezoid(values, dx=1.0))


def checkpoint_success_rates(
    training_df: pd.DataFrame,
    checkpoints: list[int] | None = None,
) -> dict[int, float]:
    """Success rate at selected training episode checkpoints."""
    checkpoints = checkpoints or CHECKPOINTS
    results: dict[int, float] = {}
    for checkpoint in checkpoints:
        subset = training_df[training_df["episode"] <= checkpoint]
        if subset.empty:
            results[checkpoint] = 0.0
        else:
            results[checkpoint] = float(subset["success"].mean())
    return results


def summarize_seed_results(
    training_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
) -> dict[str, float]:
    """Aggregate per-seed training and evaluation metrics."""
    checkpoint_rates = checkpoint_success_rates(training_df)
    return {
        "final_success_rate": float(evaluation_df["success"].mean()),
        "learning_curve_auc": learning_curve_auc(training_df["success"]),
        "avg_training_return": float(training_df["reward"].mean()),
        "avg_training_length": float(training_df["episode_length"].mean()),
        "avg_repeated_state_rate": float(training_df["repeated_state_rate"].mean()),
        "avg_unique_states": float(training_df["unique_states"].mean()),
        "training_time": float(training_df["execution_time"].sum()),
        "checkpoint_500": checkpoint_rates.get(500, 0.0),
        "checkpoint_1000": checkpoint_rates.get(1000, 0.0),
        "checkpoint_2000": checkpoint_rates.get(2000, 0.0),
        "checkpoint_5000": checkpoint_rates.get(5000, 0.0),
        "checkpoint_10000": checkpoint_rates.get(10000, 0.0),
    }


def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, and 95% CI across seeds."""
    numeric_cols = [
        col
        for col in summary_df.columns
        if col not in {"algorithm", "environment", "seed"}
    ]
    rows = []
    for (algorithm, environment), group in summary_df.groupby(["algorithm", "environment"]):
        row = {"algorithm": algorithm, "environment": environment, "n_seeds": len(group)}
        for col in numeric_cols:
            values = group[col].astype(float)
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)
