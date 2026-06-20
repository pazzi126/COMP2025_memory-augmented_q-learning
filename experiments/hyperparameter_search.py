"""Hyperparameter sensitivity experiments."""

from __future__ import annotations

import itertools
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiments import run_single_experiment
from src.utils import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, ensure_result_dirs, load_yaml


def run_hyperparameter_search() -> None:
    ensure_result_dirs()
    experiment_config = load_yaml(PROJECT_ROOT / "configs" / "experiments.yaml")
    hp_cfg = experiment_config["experiments"]["hyperparameter_search"]
    environments = hp_cfg.get("environments", [hp_cfg.get("environment", "frozenlake_4x4")])
    num_seeds = int(hp_cfg["num_seeds"])

    records: list[dict] = []

    stm_grid = list(
        itertools.product(
            hp_cfg["stm_memory_sizes"],
            hp_cfg["stm_repetition_penalties"],
        )
    )
    for environment in environments:
        for memory_size, penalty in tqdm(stm_grid, desc=f"STM hyperparameters ({environment})"):
            for seed in range(num_seeds):
                config = deepcopy(experiment_config)
                config["agents"]["stm"]["stm_memory_size"] = memory_size
                config["agents"]["stm"]["stm_repetition_penalty"] = penalty
                config["agents"]["combined"]["stm_memory_size"] = memory_size
                config["agents"]["combined"]["stm_repetition_penalty"] = penalty

                _, _, summary, _, _, _ = run_single_experiment("stm", environment, seed, config)
                summary.update(
                    {
                        "parameter": "stm",
                        "memory_size": memory_size,
                        "repetition_penalty": penalty,
                        "memory_strength": None,
                        "environment": environment,
                    }
                )
                records.append(summary)

    ltm_grid = hp_cfg["ltm_memory_strengths"]
    for environment in environments:
        for strength in tqdm(ltm_grid, desc=f"LTM hyperparameters ({environment})"):
            for seed in range(num_seeds):
                config = deepcopy(experiment_config)
                config["agents"]["ltm"]["ltm_memory_strength"] = strength
                config["agents"]["combined"]["ltm_memory_strength"] = strength

                _, _, summary, _, _, _ = run_single_experiment("ltm", environment, seed, config)
                summary.update(
                    {
                        "parameter": "ltm",
                        "memory_size": None,
                        "repetition_penalty": None,
                        "memory_strength": strength,
                        "environment": environment,
                    }
                )
                records.append(summary)

    hp_df = pd.DataFrame(records)
    hp_df.to_csv(RAW_DIR / "hyperparameter_search_results.csv", index=False)

    grouped = (
        hp_df.groupby(
            [
                "parameter",
                "environment",
                "memory_size",
                "repetition_penalty",
                "memory_strength",
            ],
            dropna=False,
        )["final_success_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "mean": "mean_success",
            "std": "std_success",
            "count": "seeds",
        }
    )
    grouped.to_csv(PROCESSED_DIR / "hyperparameter_summary.csv", index=False)
    print("Hyperparameter search complete.")


if __name__ == "__main__":
    run_hyperparameter_search()
