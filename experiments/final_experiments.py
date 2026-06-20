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
from src.metrics import (
    aggregate_summary,
    build_evaluation_summary,
    build_main_results_table,
    build_repeated_state_summary,
)
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
    all_visitation: list[pd.DataFrame] = []
    all_policies: list[pd.DataFrame] = []
    all_memory_snapshots: list[pd.DataFrame] = []
    repeat_window_size = int(experiment_config["training"].get("repeat_window_size", 10))

    combinations = list(
        itertools.product(algorithms, environments, range(num_seeds))
    )

    for algorithm, environment, seed in tqdm(combinations, desc="Final experiments"):
        (
            training_df,
            evaluation_df,
            summary,
            visitation_df,
            policy_df,
            memory_df,
        ) = run_single_experiment(
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
        all_visitation.append(visitation_df)
        all_policies.append(policy_df)
        if not memory_df.empty:
            all_memory_snapshots.append(memory_df)

    training_combined = pd.concat(all_training, ignore_index=True)
    evaluation_combined = pd.concat(all_evaluation, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    aggregated = aggregate_summary(summary_df)
    evaluation_summary = build_evaluation_summary(evaluation_combined, repeat_window_size)
    repeated_state_summary = build_repeated_state_summary(
        training_combined, evaluation_combined, repeat_window_size
    )
    main_results_table = build_main_results_table(aggregated)
    visitation_df = pd.concat(all_visitation, ignore_index=True)
    policy_df = pd.concat(all_policies, ignore_index=True)
    memory_snapshots_df = (
        pd.concat(all_memory_snapshots, ignore_index=True)
        if all_memory_snapshots
        else pd.DataFrame()
    )

    training_combined.to_csv(RAW_DIR / "all_training_results.csv", index=False)
    evaluation_combined.to_csv(RAW_DIR / "all_evaluation_results.csv", index=False)
    summary_df.to_csv(PROCESSED_DIR / "per_seed_summary.csv", index=False)
    aggregated.to_csv(PROCESSED_DIR / "aggregated_summary.csv", index=False)
    evaluation_summary.to_csv(PROCESSED_DIR / "evaluation_summary.csv", index=False)
    repeated_state_summary.to_csv(PROCESSED_DIR / "repeated_state_summary.csv", index=False)
    main_results_table.to_csv(PROCESSED_DIR / "main_results_table.csv", index=False)
    visitation_df.to_csv(PROCESSED_DIR / "state_visitation_summary.csv", index=False)
    policy_df.to_csv(PROCESSED_DIR / "policy_maps.csv", index=False)
    memory_snapshots_df.to_csv(PROCESSED_DIR / "memory_evolution.csv", index=False)

    print("Final experiments complete.")
    print(f"Training records: {len(training_combined)}")
    print(f"Evaluation records: {len(evaluation_combined)}")
    print(f"Seed summaries: {len(summary_df)}")


if __name__ == "__main__":
    run_final_experiments()
