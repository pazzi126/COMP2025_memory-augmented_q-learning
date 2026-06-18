"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Abstract tabular reinforcement-learning agent."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    @abstractmethod
    def select_action(self, state: int, epsilon: float) -> int:
        """Choose an action for the given state."""

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Apply a learning update after a transition."""

    def begin_episode(self) -> None:
        """Hook called at the start of each training episode."""

    def end_episode(self, success: bool) -> None:
        """Hook called at the end of each training episode."""

    def on_step(
        self,
        state: int,
        action: int,
        next_state: int,
        reward: float,
    ) -> float:
        """Hook called after each environment step; may adjust reward."""
        return reward

    def greedy_action(self, state: int) -> int:
        """Select the greedy action for evaluation."""
        return int(np.argmax(self.Q[state]))
