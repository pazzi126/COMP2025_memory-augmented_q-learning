"""Policy evaluation for trained agents."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments import make_environment
from src.training import create_agent
from src.utils import RepeatRateTracker, set_seed


def evaluate_agent(
    algorithm: str,
    environment_name: str,
    seed: int,
    config: dict[str, Any],
    agent: BaseAgent | None = None,
) -> tuple[list[dict[str, Any]], BaseAgent, dict[str, Any]]:
    """Evaluate a greedy policy and return per-episode evaluation records."""
    set_seed(seed)
    env = make_environment(environment_name, seed=seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    if agent is None:
        agent = create_agent(algorithm, n_states, n_actions, config)

    evaluation_episodes = int(config.get("evaluation_episodes", 500))
    max_steps = int(config.get("max_episode_steps", 200))
    repeat_window_size = int(config.get("repeat_window_size", 10))
    records: list[dict[str, Any]] = []
    repeat_tracker = RepeatRateTracker(window_size=repeat_window_size)
    state_visits = np.zeros(n_states, dtype=np.float64)

    for episode in range(evaluation_episodes):
        state, _ = env.reset(seed=seed + 100000 + episode)
        total_reward = 0.0
        steps = 0
        episode_repeats = 0
        repeat_tracker.reset(int(state))

        start_time = time.perf_counter()

        for _ in range(max_steps):
            action = agent.greedy_action(int(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if repeat_tracker.observe_next_state(int(next_state)):
                episode_repeats += 1

            total_reward += float(reward)
            steps += 1
            state = next_state
            state_visits[int(state)] += 1

            if done:
                break

        elapsed = time.perf_counter() - start_time
        success = bool(terminated and reward > 0)
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
                "epsilon": 0.0,
                "execution_time": elapsed,
            }
        )

    env.close()
    diagnostics = {
        "repeat_window_size": repeat_window_size,
        "evaluation_state_visits": state_visits,
    }
    return records, agent, diagnostics
