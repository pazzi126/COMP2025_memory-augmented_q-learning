"""Training loop for tabular agents."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments import make_environment
from src.utils import RepeatRateTracker, linear_epsilon, set_seed


def create_agent(
    algorithm: str,
    n_states: int,
    n_actions: int,
    config: dict[str, Any],
) -> BaseAgent:
    """Instantiate an agent from an algorithm name."""
    common = {
        "n_states": n_states,
        "n_actions": n_actions,
        "alpha": config.get("alpha", 0.1),
        "gamma": config.get("gamma", 0.99),
    }

    if algorithm == "qlearning":
        from src.agents.qlearning import QLearningAgent

        return QLearningAgent(**common)
    if algorithm == "stm":
        from src.agents.stm_qlearning import STMAgent

        return STMAgent(
            **common,
            memory_size=config.get("stm_memory_size", 10),
            repetition_penalty=config.get("stm_repetition_penalty", 0.05),
        )
    if algorithm == "ltm":
        from src.agents.ltm_qlearning import LTMAgent

        return LTMAgent(
            **common,
            memory_strength=config.get("ltm_memory_strength", 0.10),
            memory_decay=config.get("ltm_memory_decay", 0.0),
        )
    if algorithm == "combined":
        from src.agents.combined_qlearning import CombinedAgent

        return CombinedAgent(
            **common,
            memory_size=config.get("stm_memory_size", 10),
            repetition_penalty=config.get("stm_repetition_penalty", 0.05),
            memory_strength=config.get("ltm_memory_strength", 0.10),
            memory_decay=config.get("ltm_memory_decay", 0.0),
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def train_agent(
    algorithm: str,
    environment_name: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], BaseAgent, dict[str, Any]]:
    """Train an agent and return per-episode training records."""
    set_seed(seed)
    env = make_environment(environment_name, seed=seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = create_agent(algorithm, n_states, n_actions, config)

    training_episodes = int(config.get("training_episodes", 10000))
    max_steps = int(config.get("max_episode_steps", 200))
    epsilon_start = float(config.get("epsilon_start", 1.0))
    epsilon_min = float(config.get("epsilon_min", 0.05))
    repeat_window_size = int(config.get("repeat_window_size", 10))

    records: list[dict[str, Any]] = []
    early_cutoff = max(1, int(0.2 * training_episodes))
    late_cutoff = max(1, int(0.8 * training_episodes))
    early_state_visits = np.zeros(n_states, dtype=np.float64)
    late_state_visits = np.zeros(n_states, dtype=np.float64)
    memory_snapshots: dict[int, np.ndarray] = {}
    snapshot_episodes = set(config.get("memory_snapshot_episodes", [1000, 5000, 10000]))
    repeat_tracker = RepeatRateTracker(window_size=repeat_window_size)

    for episode in range(training_episodes):
        epsilon = linear_epsilon(episode, training_episodes, epsilon_start, epsilon_min)
        state, _ = env.reset(seed=seed + episode)
        agent.begin_episode()
        repeat_tracker.reset(int(state))

        total_reward = 0.0
        steps = 0
        visited_states = {int(state)}
        episode_repeats = 0

        start_time = time.perf_counter()

        for _ in range(max_steps):
            action = agent.select_action(int(state), epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            adjusted_reward = agent.on_step(
                int(state),
                int(action),
                int(next_state),
                float(reward),
            )
            agent.update(
                int(state),
                int(action),
                adjusted_reward,
                int(next_state),
                done,
            )

            is_repeat = repeat_tracker.observe_next_state(int(next_state))
            if is_repeat:
                episode_repeats += 1

            total_reward += float(reward)
            steps += 1
            state = next_state
            visited_states.add(int(state))
            if episode + 1 <= early_cutoff:
                early_state_visits[int(state)] += 1
            if episode + 1 > late_cutoff:
                late_state_visits[int(state)] += 1

            if done:
                break

        elapsed = time.perf_counter() - start_time
        success = bool(terminated and reward > 0)
        agent.end_episode(success)

        repeated_state_rate = episode_repeats / steps if steps > 0 else 0.0

        records.append(
            {
                "algorithm": algorithm,
                "environment": environment_name,
                "seed": seed,
                "episode": episode + 1,
                "reward": total_reward,
                "success": int(success),
                "episode_length": steps,
                "repeated_state_count": episode_repeats,
                "repeated_state_rate": repeated_state_rate,
                "unique_states": len(visited_states),
                "epsilon": epsilon,
                "execution_time": elapsed,
            }
        )
        if hasattr(agent, "long_term") and (episode + 1) in snapshot_episodes:
            memory_snapshots[episode + 1] = np.copy(agent.long_term.memory)

    env.close()
    policy_actions = [agent.greedy_action(state_idx) for state_idx in range(n_states)]
    diagnostics = {
        "repeat_window_size": repeat_window_size,
        "early_state_visits": early_state_visits,
        "late_state_visits": late_state_visits,
        "policy_actions": policy_actions,
        "memory_snapshots": memory_snapshots,
    }
    return records, agent, diagnostics
