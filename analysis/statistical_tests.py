"""Statistical comparisons between agents."""

from __future__ import annotations

import itertools
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

    algorithms = ["qlearning", "stm", "ltm", "combined"]
    environments = summary_df["environment"].unique()
    metric = "final_success_rate"
    rows = []
    p_values = []

    for environment in environments:
        env_data = summary_df[summary_df["environment"] == environment]
        for algo_a, algo_b in itertools.combinations(algorithms, 2):
            data_a = env_data[env_data["algorithm"] == algo_a].sort_values("seed")
            data_b = env_data[env_data["algorithm"] == algo_b].sort_values("seed")
            x = data_a[metric].to_numpy()
            y = data_b[metric].to_numpy()

            if len(x) != len(y):
                continue

            statistic, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
            effect_size = rank_biserial_effect_size(x, y)
            rows.append(
                {
                    "environment": environment,
                    "algorithm_a": algo_a,
                    "algorithm_b": algo_b,
                    "metric": metric,
                    "mean_a": float(x.mean()),
                    "mean_b": float(y.mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value_raw": float(p_value),
                    "effect_size": float(effect_size),
                }
            )
            p_values.append(float(p_value))

    corrected = holm_correction(p_values)
    for row, corrected_p in zip(rows, corrected):
        row["p_value_holm"] = corrected_p

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(PROCESSED_DIR / "statistical_comparisons.csv", index=False)
    print(f"Statistical comparisons saved to {PROCESSED_DIR / 'statistical_comparisons.csv'}")


if __name__ == "__main__":
    run_statistical_tests()
