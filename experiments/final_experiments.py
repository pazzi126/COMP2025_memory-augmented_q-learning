"""Run all final experiments across agents, environments, and seeds."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiments import run_single_experiment
from src.metrics import aggregate_summary
from src.utils import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, ensure_result_dirs, load_yaml


def run_final_experiments() -> None:
    ensure_result_dirs()
    experiment_config = load_yaml(PROJECT_ROOT / "configs" / "experiments.yaml")
    main_cfg = experiment_config["experiments"]["main_comparison"]
    algorithms = main_cfg["algorithms"]
    environments = main_cfg["environments"]
    num_seeds = int(experiment_config["training"]["num_seeds"])

    all_training: list[pd.DataFrame] = []
    all_evaluation: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    combinations = list(
        itertools.product(algorithms, environments, range(num_seeds))
    )

    for algorithm, environment, seed in tqdm(combinations, desc="Final experiments"):
        training_df, evaluation_df, summary = run_single_experiment(
            algorithm,
            environment,
            seed,
            experiment_config,
        )
        prefix = f"{algorithm}_{environment}_seed{seed}"
        training_df.to_csv(RAW_DIR / f"{prefix}_training.csv", index=False)
        evaluation_df.to_csv(RAW_DIR / f"{prefix}_evaluation.csv", index=False)

        all_training.append(training_df)
        all_evaluation.append(evaluation_df)
        all_summaries.append(summary)

    training_combined = pd.concat(all_training, ignore_index=True)
    evaluation_combined = pd.concat(all_evaluation, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    aggregated = aggregate_summary(summary_df)

    training_combined.to_csv(RAW_DIR / "all_training_results.csv", index=False)
    evaluation_combined.to_csv(RAW_DIR / "all_evaluation_results.csv", index=False)
    summary_df.to_csv(PROCESSED_DIR / "per_seed_summary.csv", index=False)
    aggregated.to_csv(PROCESSED_DIR / "aggregated_summary.csv", index=False)

    print("Final experiments complete.")
    print(f"Training records: {len(training_combined)}")
    print(f"Evaluation records: {len(evaluation_combined)}")
    print(f"Seed summaries: {len(summary_df)}")


if __name__ == "__main__":
    run_final_experiments()
