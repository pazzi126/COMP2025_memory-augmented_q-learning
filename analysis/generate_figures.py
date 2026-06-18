"""Generate figures from experiment results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import FIGURES_DIR, PROCESSED_DIR, RAW_DIR, ensure_result_dirs


AGENT_LABELS = {
    "qlearning": "Q-Learning",
    "stm": "STM",
    "ltm": "LTM",
    "combined": "Combined",
}

ENV_LABELS = {
    "frozenlake_4x4": "FL 4x4",
    "frozenlake_8x8": "FL 8x8",
    "frozenlake_8x8_stochastic": "FL 8x8 Stoch.",
    "custom_loop": "Custom Loop",
}

COLORS = {
    "qlearning": "#1f77b4",
    "stm": "#ff7f0e",
    "ltm": "#2ca02c",
    "combined": "#d62728",
}


def _load_training_data() -> pd.DataFrame:
    path = RAW_DIR / "all_training_results.csv"
    if not path.exists():
        raise FileNotFoundError("Run experiments/final_experiments.py first.")
    return pd.read_csv(path)


def _load_summary_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "per_seed_summary.csv"
    if not path.exists():
        raise FileNotFoundError("Run experiments/final_experiments.py first.")
    return pd.read_csv(path)


def plot_learning_curves(training_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 2: training success rate across episodes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    environments = training_df["environment"].unique()

    for ax, environment in zip(axes, environments):
        env_data = training_df[training_df["environment"] == environment]
        for algorithm in ["qlearning", "stm", "ltm", "combined"]:
            algo_data = env_data[env_data["algorithm"] == algorithm]
            grouped = (
                algo_data.groupby("episode")["success"]
                .agg(["mean", "std"])
                .reset_index()
            )
            episodes = grouped["episode"]
            mean = grouped["mean"]
            std = grouped["std"]
            ax.plot(
                episodes,
                mean,
                label=AGENT_LABELS[algorithm],
                color=COLORS[algorithm],
                linewidth=1.5,
            )
            ax.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=COLORS[algorithm],
                alpha=0.15,
            )
        ax.set_title(ENV_LABELS.get(environment, environment))
        ax.set_xlim(1, training_df["episode"].max())
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Training Episode")
    axes[0].set_ylabel("Success Rate")
    axes[2].set_ylabel("Success Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Training Success Rate Across Episodes", fontsize=14, y=1.04)
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_success_rates(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 3: final success rate across environments."""
    environments = summary_df["environment"].unique()
    algorithms = ["qlearning", "stm", "ltm", "combined"]
    x = np.arange(len(environments))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, algorithm in enumerate(algorithms):
        means = []
        stds = []
        for environment in environments:
            subset = summary_df[
                (summary_df["algorithm"] == algorithm)
                & (summary_df["environment"] == environment)
            ]
            means.append(subset["final_success_rate"].mean())
            stds.append(subset["final_success_rate"].std(ddof=1))
        ax.bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=AGENT_LABELS[algorithm],
            color=COLORS[algorithm],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(env, env) for env in environments])
    ax.set_ylabel("Final Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Final Success Rate Across Environments")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_final_success_rates.png", dpi=150)
    plt.close(fig)


def plot_repeated_state_rates(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 4: repeated-state rate."""
    environments = summary_df["environment"].unique()
    algorithms = ["qlearning", "stm", "ltm", "combined"]
    x = np.arange(len(environments))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, algorithm in enumerate(algorithms):
        means = []
        stds = []
        for environment in environments:
            subset = summary_df[
                (summary_df["algorithm"] == algorithm)
                & (summary_df["environment"] == environment)
            ]
            means.append(subset["avg_repeated_state_rate"].mean())
            stds.append(subset["avg_repeated_state_rate"].std(ddof=1))
        ax.bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=AGENT_LABELS[algorithm],
            color=COLORS[algorithm],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(env, env) for env in environments])
    ax.set_ylabel("Repeated-State Rate")
    ax.set_title("Repeated-State Rate Across Environments")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_repeated_state_rates.png", dpi=150)
    plt.close(fig)


def plot_episode_lengths(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Episode length comparison."""
    environments = summary_df["environment"].unique()
    algorithms = ["qlearning", "stm", "ltm", "combined"]
    x = np.arange(len(environments))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, algorithm in enumerate(algorithms):
        means = []
        stds = []
        for environment in environments:
            subset = summary_df[
                (summary_df["algorithm"] == algorithm)
                & (summary_df["environment"] == environment)
            ]
            means.append(subset["avg_training_length"].mean())
            stds.append(subset["avg_training_length"].std(ddof=1))
        ax.bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=AGENT_LABELS[algorithm],
            color=COLORS[algorithm],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(env, env) for env in environments])
    ax.set_ylabel("Average Episode Length")
    ax.set_title("Average Training Episode Length")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure5_episode_lengths.png", dpi=150)
    plt.close(fig)


def plot_hyperparameter_sensitivity(output_dir: Path) -> None:
    """Figure 6: hyperparameter sensitivity analysis."""
    path = PROCESSED_DIR / "hyperparameter_summary.csv"
    if not path.exists():
        return

    hp_df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stm_df = hp_df[hp_df["parameter"] == "stm"].copy()
    if not stm_df.empty:
        stm_df["label"] = stm_df.apply(
            lambda row: f"size={int(row['memory_size'])}, pen={row['repetition_penalty']}",
            axis=1,
        )
        axes[0].bar(stm_df["label"], stm_df["mean"], yerr=stm_df["std"], capsize=3)
        axes[0].set_title("STM Hyperparameter Sensitivity")
        axes[0].set_ylabel("Final Success Rate")
        axes[0].tick_params(axis="x", rotation=45)

    ltm_df = hp_df[hp_df["parameter"] == "ltm"].copy()
    if not ltm_df.empty:
        ltm_df["label"] = ltm_df["memory_strength"].apply(lambda value: f"strength={value}")
        axes[1].bar(ltm_df["label"], ltm_df["mean"], yerr=ltm_df["std"], capsize=3)
        axes[1].set_title("LTM Hyperparameter Sensitivity")
        axes[1].set_ylabel("Final Success Rate")

    fig.tight_layout()
    fig.savefig(output_dir / "figure6_hyperparameter_sensitivity.png", dpi=150)
    plt.close(fig)


def generate_all_figures() -> None:
    ensure_result_dirs()
    training_df = _load_training_data()
    summary_df = _load_summary_data()

    plot_learning_curves(training_df, FIGURES_DIR)
    plot_final_success_rates(summary_df, FIGURES_DIR)
    plot_repeated_state_rates(summary_df, FIGURES_DIR)
    plot_episode_lengths(summary_df, FIGURES_DIR)
    plot_hyperparameter_sensitivity(FIGURES_DIR)
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
