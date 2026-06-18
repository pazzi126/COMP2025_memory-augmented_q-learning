"""Environment factory helpers."""

from __future__ import annotations

import gymnasium as gym

from src.environments.custom_maps import CUSTOM_LOOP_MAP


def make_environment(environment_name: str, seed: int | None = None):
    """Create a Gymnasium environment from a project environment name."""
    if environment_name == "frozenlake_4x4":
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    elif environment_name == "frozenlake_8x8":
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    elif environment_name == "frozenlake_8x8_stochastic":
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    elif environment_name == "custom_loop":
        env = gym.make(
            "FrozenLake-v1",
            desc=CUSTOM_LOOP_MAP,
            is_slippery=False,
        )
    else:
        raise ValueError(f"Unknown environment: {environment_name}")

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
