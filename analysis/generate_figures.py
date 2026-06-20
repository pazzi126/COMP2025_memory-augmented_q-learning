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
ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}
MAP_LAYOUTS = {
    "frozenlake_4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "frozenlake_8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "frozenlake_8x8_stochastic": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "custom_loop": [
        "SFFFFFFF",
        "FHFHFFHF",
        "FFFHFFHF",
        "FHHFFFHF",
        "FFFFHFFF",
        "HFHFFFHF",
        "FFHFFHFF",
        "FFFHFFFG",
    ],
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


def plot_learning_curves(
    training_df: pd.DataFrame, output_dir: Path, window_size: int = 200
) -> None:
    """Figure 2: rolling learning curves with 95% CI."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    environments = [
        "frozenlake_4x4",
        "frozenlake_8x8",
        "frozenlake_8x8_stochastic",
        "custom_loop",
    ]

    for ax, environment in zip(axes, environments):
        env_data = training_df[training_df["environment"] == environment]
        for algorithm in ["qlearning", "stm", "ltm", "combined"]:
            algo_data = env_data[env_data["algorithm"] == algorithm].sort_values(
                ["seed", "episode"]
            )
            if algo_data.empty:
                continue
            algo_data = algo_data.copy()
            algo_data["rolling_success"] = algo_data.groupby("seed")["success"].transform(
                lambda values: values.rolling(window=window_size, min_periods=1).mean()
            )
            grouped = (
                algo_data.groupby("episode")["rolling_success"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            episodes = grouped["episode"]
            mean = grouped["mean"]
            ci95 = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])
            ax.plot(
                episodes,
                mean,
                label=AGENT_LABELS[algorithm],
                color=COLORS[algorithm],
                linewidth=1.5,
            )
            ax.fill_between(
                episodes,
                mean - ci95,
                mean + ci95,
                color=COLORS[algorithm],
                alpha=0.15,
            )
        ax.set_title(ENV_LABELS.get(environment, environment))
        ax.set_xlim(1, training_df["episode"].max())
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Training Episode")
    axes[0].set_ylabel("Rolling Success Rate")
    axes[2].set_ylabel("Rolling Success Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Rolling Training Success (Window=200) with 95% CI", fontsize=14, y=1.04)
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


def plot_repeated_state_rates(repeated_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 4: corrected repeated-state rate and delta vs baseline."""
    environments = repeated_df["environment"].unique()
    algorithms = ["qlearning", "stm", "ltm", "combined"]
    x = np.arange(len(environments))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for index, algorithm in enumerate(algorithms):
        means = []
        stds = []
        for environment in environments:
            subset = repeated_df[
                (repeated_df["algorithm"] == algorithm)
                & (repeated_df["environment"] == environment)
            ]
            means.append(subset["evaluation_repeat_rate_mean"].mean())
            stds.append(subset["evaluation_repeat_rate_std"].fillna(0.0).mean())
        axes[0].bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=AGENT_LABELS[algorithm],
            color=COLORS[algorithm],
            alpha=0.9,
        )

    baseline = (
        repeated_df[repeated_df["algorithm"] == "qlearning"]
        .set_index("environment")["evaluation_repeat_rate_mean"]
        .to_dict()
    )
    for index, algorithm in enumerate(["stm", "ltm", "combined"]):
        deltas = []
        for environment in environments:
            method_rate = repeated_df[
                (repeated_df["algorithm"] == algorithm)
                & (repeated_df["environment"] == environment)
            ]["evaluation_repeat_rate_mean"].mean()
            deltas.append(method_rate - baseline.get(environment, 0.0))
        axes[1].bar(
            x + (index - 1.0) * width,
            deltas,
            width,
            label=f"{AGENT_LABELS[algorithm]} - Baseline",
            color=COLORS[algorithm],
            alpha=0.9,
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ENV_LABELS.get(env, env) for env in environments], rotation=20)
    axes[0].set_ylabel("Evaluation Repeated-State Rate")
    axes[0].set_title("Corrected Repeated-State Rate")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ENV_LABELS.get(env, env) for env in environments], rotation=20)
    axes[1].set_title("Delta vs Baseline")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_repeated_state_rates.png", dpi=150)
    fig.savefig(output_dir / "figure_repeated_state_rate_corrected.png", dpi=150)
    plt.close(fig)


def plot_episode_lengths(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Success-conditioned evaluation episode-length comparison."""
    environments = summary_df["environment"].unique()
    algorithms = ["qlearning", "stm", "ltm", "combined"]
    x = np.arange(len(environments))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    metric_map = [
        ("evaluation_success_length_mean", "Successful Episodes"),
        ("evaluation_failed_length_mean", "Failed Episodes"),
    ]

    for ax, (metric, title) in zip(axes, metric_map):
        for index, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for environment in environments:
                subset = summary_df[
                    (summary_df["algorithm"] == algorithm)
                    & (summary_df["environment"] == environment)
                ]
                means.append(subset[metric].mean())
                stds.append(subset[metric].std(ddof=1))
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
        ax.set_xticklabels([ENV_LABELS.get(env, env) for env in environments], rotation=20)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Average Episode Length")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_dir / "figure5_episode_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hyperparameter_sensitivity(output_dir: Path) -> None:
    """Figure 6 + sensitivity plots for STM and LTM."""
    path = PROCESSED_DIR / "hyperparameter_summary.csv"
    if not path.exists():
        return

    hp_df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stm_df = hp_df[hp_df["parameter"] == "stm"].copy()
    if not stm_df.empty:
        pivot = stm_df.pivot_table(
            index="repetition_penalty",
            columns="memory_size",
            values="mean_success",
            aggfunc="mean",
        )
        im = axes[0].imshow(pivot.values, cmap="viridis", aspect="auto")
        axes[0].set_xticks(np.arange(len(pivot.columns)))
        axes[0].set_xticklabels([int(value) for value in pivot.columns])
        axes[0].set_yticks(np.arange(len(pivot.index)))
        axes[0].set_yticklabels([f"{value:.2f}" for value in pivot.index])
        axes[0].set_xlabel("STM Memory Size")
        axes[0].set_ylabel("STM Repetition Penalty")
        axes[0].set_title("STM Sensitivity Heatmap")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    ltm_df = hp_df[hp_df["parameter"] == "ltm"].copy()
    if not ltm_df.empty:
        for environment, subset in ltm_df.groupby("environment"):
            subset = subset.sort_values("memory_strength")
            axes[1].plot(
                subset["memory_strength"],
                subset["mean_success"],
                marker="o",
                label=ENV_LABELS.get(environment, environment),
            )
        axes[1].set_xlabel("LTM Memory Strength")
        axes[1].set_ylabel("Final Success Rate")
        axes[1].set_title("LTM Sensitivity Curve")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "figure6_hyperparameter_sensitivity.png", dpi=150)
    fig.savefig(output_dir / "figure_stm_sensitivity_heatmap.png", dpi=150)
    fig.savefig(output_dir / "figure_ltm_sensitivity_curve.png", dpi=150)
    plt.close(fig)


def plot_seed_distributions(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplots showing per-seed final success distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    environments = [
        "frozenlake_4x4",
        "frozenlake_8x8",
        "frozenlake_8x8_stochastic",
        "custom_loop",
    ]
    algorithms = ["qlearning", "stm", "ltm", "combined"]
    for ax, environment in zip(axes, environments):
        env_data = summary_df[summary_df["environment"] == environment]
        data = [env_data[env_data["algorithm"] == algo]["final_success_rate"] for algo in algorithms]
        box = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=[AGENT_LABELS[a] for a in algorithms],
        )
        for patch, algo in zip(box["boxes"], algorithms):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.65)
        ax.set_title(ENV_LABELS.get(environment, environment))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Final Success Rate")
    axes[2].set_ylabel("Final Success Rate")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_seed_distributions.png", dpi=150)
    plt.close(fig)


def plot_state_visitation_heatmaps(output_dir: Path) -> None:
    """Heatmaps for early/late/evaluation state visitation frequencies."""
    path = PROCESSED_DIR / "state_visitation_summary.csv"
    if not path.exists():
        return
    visit_df = pd.read_csv(path)
    for environment, env_group in visit_df.groupby("environment"):
        for phase, phase_group in env_group.groupby("phase"):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, algorithm in zip(axes, ["qlearning", "stm", "ltm", "combined"]):
                subset = phase_group[phase_group["algorithm"] == algorithm]
                if subset.empty:
                    ax.axis("off")
                    continue
                avg_visits = (
                    subset.groupby("state")["normalized_visits"].mean().sort_index().to_numpy()
                )
                side = int(np.sqrt(len(avg_visits)))
                if side * side != len(avg_visits):
                    side = len(avg_visits)
                    heatmap = avg_visits.reshape(1, -1)
                else:
                    heatmap = avg_visits.reshape(side, side)
                image = ax.imshow(heatmap, cmap="magma")
                ax.set_title(AGENT_LABELS[algorithm])
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{ENV_LABELS.get(environment, environment)} - {phase}")
            fig.tight_layout()
            fig.savefig(
                output_dir / f"figure_heatmap_{environment}_{phase}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_policy_maps(output_dir: Path) -> None:
    """Policy maps based on most frequent greedy action across seeds."""
    path = PROCESSED_DIR / "policy_maps.csv"
    if not path.exists():
        return
    policy_df = pd.read_csv(path)
    for environment, env_group in policy_df.groupby("environment"):
        if environment not in MAP_LAYOUTS:
            continue
        layout = MAP_LAYOUTS[environment]
        n_rows = len(layout)
        n_cols = len(layout[0])
        policy_majority = (
            env_group.groupby(["algorithm", "state"])["action"]
            .agg(lambda values: int(values.mode().iloc[0]))
            .reset_index()
        )
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, algorithm in zip(axes, ["qlearning", "stm", "ltm", "combined"]):
            grid_text = []
            subset = policy_majority[policy_majority["algorithm"] == algorithm]
            action_map = dict(zip(subset["state"], subset["action"]))
            for row in range(n_rows):
                line = []
                for col in range(n_cols):
                    idx = row * n_cols + col
                    tile = layout[row][col]
                    if tile in {"H", "G"}:
                        line.append(tile)
                    elif tile == "S":
                        line.append("S")
                    else:
                        line.append(ACTION_ARROWS.get(action_map.get(idx, 2), "→"))
                grid_text.append(" ".join(line))
            ax.text(0.02, 0.98, "\n".join(grid_text), va="top", family="monospace", fontsize=11)
            ax.set_title(AGENT_LABELS[algorithm])
            ax.axis("off")
        fig.suptitle(f"Policy Maps - {ENV_LABELS.get(environment, environment)}")
        fig.tight_layout()
        fig.savefig(output_dir / f"figure_policy_map_{environment}.png", dpi=150)
        plt.close(fig)


def plot_memory_evolution(output_dir: Path) -> None:
    """Visualize LTM memory evolution across checkpoints."""
    path = PROCESSED_DIR / "memory_evolution.csv"
    if not path.exists():
        return
    memory_df = pd.read_csv(path)
    if memory_df.empty:
        return
    memory_df = memory_df[memory_df["algorithm"].isin(["ltm", "combined"])]
    for (environment, algorithm), subset in memory_df.groupby(["environment", "algorithm"]):
        checkpoints = sorted(subset["episode_checkpoint"].unique())
        fig, axes = plt.subplots(1, len(checkpoints), figsize=(5 * len(checkpoints), 4))
        if len(checkpoints) == 1:
            axes = [axes]
        for ax, checkpoint in zip(axes, checkpoints):
            snap = subset[subset["episode_checkpoint"] == checkpoint]
            matrix = (
                snap.groupby(["state", "action"])["memory_value"].mean().unstack(fill_value=0.0)
            )
            image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
            ax.set_title(f"Episode {checkpoint}")
            ax.set_xlabel("Action")
            ax.set_ylabel("State")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Memory Evolution - {AGENT_LABELS.get(algorithm, algorithm)} - {ENV_LABELS.get(environment, environment)}")
        fig.tight_layout()
        fig.savefig(output_dir / f"figure_memory_evolution_{algorithm}_{environment}.png", dpi=150)
        plt.close(fig)


def generate_all_figures() -> None:
    ensure_result_dirs()
    training_df = _load_training_data()
    summary_df = _load_summary_data()
    repeated_path = PROCESSED_DIR / "repeated_state_summary.csv"
    repeated_df = pd.read_csv(repeated_path) if repeated_path.exists() else pd.DataFrame()

    plot_learning_curves(training_df, FIGURES_DIR)
    plot_final_success_rates(summary_df, FIGURES_DIR)
    if not repeated_df.empty:
        plot_repeated_state_rates(repeated_df, FIGURES_DIR)
    plot_episode_lengths(summary_df, FIGURES_DIR)
    plot_seed_distributions(summary_df, FIGURES_DIR)
    plot_hyperparameter_sensitivity(FIGURES_DIR)
    plot_state_visitation_heatmaps(FIGURES_DIR)
    plot_policy_maps(FIGURES_DIR)
    plot_memory_evolution(FIGURES_DIR)
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
