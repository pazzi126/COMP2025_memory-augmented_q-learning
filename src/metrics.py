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


def rolling_success_auc(episode_success: pd.Series, window_size: int = 200) -> float:
    """Area under rolling success-rate curve."""
    rolling = (
        episode_success.astype(float).rolling(window=window_size, min_periods=1).mean()
    )
    if rolling.empty:
        return 0.0
    return float(np.trapezoid(rolling.to_numpy(), dx=1.0))


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
    rolling_window_size: int = 200,
) -> dict[str, float]:
    """Aggregate per-seed training and evaluation metrics."""
    checkpoint_rates = checkpoint_success_rates(training_df)
    success_eval = evaluation_df[evaluation_df["success"] == 1]
    failed_eval = evaluation_df[evaluation_df["success"] == 0]
    rolling_auc = rolling_success_auc(
        training_df["success"], window_size=rolling_window_size
    )
    normalized_rolling_auc = (
        rolling_auc / len(training_df) if len(training_df) > 0 else 0.0
    )
    return {
        "final_success_rate": float(evaluation_df["success"].mean()),
        "learning_curve_auc": learning_curve_auc(training_df["success"]),
        "rolling_auc": rolling_auc,
        "normalized_rolling_auc": float(normalized_rolling_auc),
        "avg_training_return": float(training_df["reward"].mean()),
        "avg_training_length": float(training_df["episode_length"].mean()),
        "avg_repeated_state_rate": float(training_df["repeated_state_rate"].mean()),
        "avg_unique_states": float(training_df["unique_states"].mean()),
        "training_time": float(training_df["execution_time"].sum()),
        "evaluation_return_mean": float(evaluation_df["reward"].mean()),
        "evaluation_episode_length_mean": float(evaluation_df["episode_length"].mean()),
        "evaluation_success_length_mean": float(
            success_eval["episode_length"].mean() if not success_eval.empty else 0.0
        ),
        "evaluation_failed_length_mean": float(
            failed_eval["episode_length"].mean() if not failed_eval.empty else 0.0
        ),
        "evaluation_repeated_state_rate_mean": float(
            evaluation_df["repeated_state_rate"].mean()
        ),
        "evaluation_inference_time_mean": float(evaluation_df["execution_time"].mean()),
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


def build_evaluation_summary(
    evaluation_df: pd.DataFrame, repeat_window_size: int
) -> pd.DataFrame:
    """Aggregate evaluation metrics by algorithm and environment."""
    rows = []
    for (algorithm, environment), group in evaluation_df.groupby(["algorithm", "environment"]):
        success_group = group[group["success"] == 1]
        failed_group = group[group["success"] == 0]
        rows.append(
            {
                "algorithm": algorithm,
                "environment": environment,
                "repeat_window_size": repeat_window_size,
                "evaluation_success_rate": float(group["success"].mean()),
                "evaluation_successful_episode_length": float(
                    success_group["episode_length"].mean() if not success_group.empty else 0.0
                ),
                "evaluation_failed_episode_length": float(
                    failed_group["episode_length"].mean() if not failed_group.empty else 0.0
                ),
                "evaluation_repeated_state_rate": float(group["repeated_state_rate"].mean()),
                "evaluation_return": float(group["reward"].mean()),
                "evaluation_inference_time": float(group["execution_time"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_repeated_state_summary(
    training_df: pd.DataFrame, evaluation_df: pd.DataFrame, repeat_window_size: int
) -> pd.DataFrame:
    """Create corrected repeated-state summary for training and evaluation."""
    train_rates = (
        training_df.groupby(["algorithm", "environment"])["repeated_state_rate"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "training_repeat_rate_mean", "std": "training_repeat_rate_std"})
    )
    eval_rates = (
        evaluation_df.groupby(["algorithm", "environment"])["repeated_state_rate"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "evaluation_repeat_rate_mean",
                "std": "evaluation_repeat_rate_std",
            }
        )
    )
    merged = train_rates.merge(eval_rates, on=["algorithm", "environment"], how="inner")
    merged["repeat_window_size"] = repeat_window_size
    return merged


def build_main_results_table(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact report table from wide aggregated summary."""
    columns = [
        "algorithm",
        "environment",
        "final_success_rate_mean",
        "final_success_rate_std",
        "final_success_rate_ci95",
        "normalized_rolling_auc_mean",
        "evaluation_episode_length_mean_mean",
        "evaluation_repeated_state_rate_mean_mean",
    ]
    main_df = aggregated_df[columns].copy()
    main_df = main_df.rename(
        columns={
            "normalized_rolling_auc_mean": "normalized_rolling_auc",
            "evaluation_episode_length_mean_mean": "evaluation_episode_length_mean",
            "evaluation_repeated_state_rate_mean_mean": "evaluation_repeated_state_rate_mean",
        }
    )
    return main_df
