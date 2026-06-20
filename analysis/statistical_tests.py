"""Statistical comparisons between baseline and memory agents."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PROCESSED_DIR, ensure_result_dirs


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to p-values."""
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    corrected = [1.0] * len(p_values)
    m = len(p_values)
    for rank, (index, p_value) in enumerate(indexed, start=1):
        corrected[index] = min(1.0, p_value * (m - rank + 1))
    return corrected


def rank_biserial_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank-biserial correlation as an effect size for paired samples."""
    differences = x - y
    non_zero = differences[differences != 0]
    if len(non_zero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(non_zero))
    positive = non_zero > 0
    r_plus = ranks[positive].sum()
    n = len(non_zero)
    return (2 * r_plus) / (n * (n + 1)) - 1


def run_statistical_tests() -> None:
    ensure_result_dirs()
    summary_path = PROCESSED_DIR / "per_seed_summary.csv"
    summary_df = pd.read_csv(summary_path)
    environments = summary_df["environment"].unique()
    metrics = [
        "final_success_rate",
        "rolling_auc",
        "evaluation_repeated_state_rate_mean",
        "evaluation_success_length_mean",
        "evaluation_failed_length_mean",
    ]
    comparison_pairs = [
        ("stm", "qlearning"),
        ("ltm", "qlearning"),
        ("combined", "qlearning"),
        ("combined", "stm"),
        ("combined", "ltm"),
    ]
    rows = []
    p_values = []

    for environment in environments:
        env_data = summary_df[summary_df["environment"] == environment]
        for metric in metrics:
            for method_a, method_b in comparison_pairs:
                data_a = env_data[env_data["algorithm"] == method_a].sort_values("seed")
                data_b = env_data[env_data["algorithm"] == method_b].sort_values("seed")
                x = data_a[metric].to_numpy()
                y = data_b[metric].to_numpy()

                if len(x) != len(y) or len(x) == 0:
                    continue

                try:
                    statistic, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
                    p_value = float(p_value) if np.isfinite(p_value) else 1.0
                    statistic = float(statistic)
                except ValueError:
                    statistic = 0.0
                    p_value = 1.0

                effect_size = rank_biserial_effect_size(x, y)
                rows.append(
                    {
                        "environment": environment,
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "mean_difference": float((x - y).mean()),
                        "wilcoxon_statistic": statistic,
                        "raw_p_value": p_value,
                        "effect_size": float(effect_size),
                    }
                )
                p_values.append(p_value)

    corrected = holm_correction(p_values)
    for row, corrected_p in zip(rows, corrected):
        row["adjusted_p_value"] = corrected_p
        row["significant"] = bool(corrected_p < 0.05)

    paired_df = pd.DataFrame(rows)
    paired_df.to_csv(PROCESSED_DIR / "paired_comparisons.csv", index=False)
    paired_df.to_csv(PROCESSED_DIR / "statistical_comparisons.csv", index=False)
    print(f"Paired comparisons saved to {PROCESSED_DIR / 'paired_comparisons.csv'}")


if __name__ == "__main__":
    run_statistical_tests()
