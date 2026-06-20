"""Run a single experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import evaluate_agent
from src.metrics import summarize_seed_results
from src.training import train_agent
from src.utils import PROJECT_ROOT, RAW_DIR, ensure_result_dirs, load_yaml


def build_config(algorithm: str, experiment_config: dict) -> dict:
    """Merge global training settings with agent-specific hyperparameters."""
    config = dict(experiment_config["training"])
    config.update(experiment_config["agents"].get(algorithm, {}))
    return config


def run_single_experiment(
    algorithm: str,
    environment: str,
    seed: int,
    experiment_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train and evaluate one agent seed."""
    config = build_config(algorithm, experiment_config)
    training_records, agent, train_diag = train_agent(algorithm, environment, seed, config)
    evaluation_records, _, eval_diag = evaluate_agent(
        algorithm,
        environment,
        seed,
        config,
        agent=agent,
    )
    training_df = pd.DataFrame(training_records)
    evaluation_df = pd.DataFrame(evaluation_records)
    rolling_window_size = int(config.get("rolling_window_size", 200))
    summary = summarize_seed_results(
        training_df,
        evaluation_df,
        rolling_window_size=rolling_window_size,
    )
    summary.update(
        {
            "algorithm": algorithm,
            "environment": environment,
            "seed": seed,
        }
    )
    visitation_rows = []
    for phase, counts in {
        "early_training": train_diag["early_state_visits"],
        "late_training": train_diag["late_state_visits"],
        "evaluation": eval_diag["evaluation_state_visits"],
    }.items():
        total_visits = float(counts.sum())
        for state_idx, count in enumerate(counts):
            visitation_rows.append(
                {
                    "algorithm": algorithm,
                    "environment": environment,
                    "seed": seed,
                    "phase": phase,
                    "state": state_idx,
                    "visits": float(count),
                    "normalized_visits": float(count / total_visits) if total_visits > 0 else 0.0,
                }
            )
    visitation_df = pd.DataFrame(visitation_rows)

    policy_df = pd.DataFrame(
        {
            "algorithm": algorithm,
            "environment": environment,
            "seed": seed,
            "state": list(range(len(train_diag["policy_actions"]))),
            "action": train_diag["policy_actions"],
        }
    )

    memory_rows = []
    for checkpoint, matrix in train_diag["memory_snapshots"].items():
        for state_idx in range(matrix.shape[0]):
            for action_idx in range(matrix.shape[1]):
                memory_rows.append(
                    {
                        "algorithm": algorithm,
                        "environment": environment,
                        "seed": seed,
                        "episode_checkpoint": checkpoint,
                        "state": state_idx,
                        "action": action_idx,
                        "memory_value": float(matrix[state_idx, action_idx]),
                    }
                )
    memory_df = pd.DataFrame(memory_rows)
    return training_df, evaluation_df, summary, visitation_df, policy_df, memory_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run memory-augmented Q-learning experiments.")
    parser.add_argument("--agent", required=True, choices=["qlearning", "stm", "ltm", "combined"])
    parser.add_argument(
        "--environment",
        required=True,
        choices=[
            "frozenlake_4x4",
            "frozenlake_8x8",
            "frozenlake_8x8_stochastic",
            "custom_loop",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_result_dirs()
    experiment_config = load_yaml(PROJECT_ROOT / "configs" / "experiments.yaml")
    training_df, evaluation_df, summary, _, _, _ = run_single_experiment(
        args.agent,
        args.environment,
        args.seed,
        experiment_config,
    )

    prefix = f"{args.agent}_{args.environment}_seed{args.seed}"
    training_df.to_csv(RAW_DIR / f"{prefix}_training.csv", index=False)
    evaluation_df.to_csv(RAW_DIR / f"{prefix}_evaluation.csv", index=False)
    pd.DataFrame([summary]).to_csv(RAW_DIR / f"{prefix}_summary.csv", index=False)
    print(f"Saved results for {prefix}")


if __name__ == "__main__":
    main()
