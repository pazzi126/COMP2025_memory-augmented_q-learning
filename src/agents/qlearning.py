# AI-assisted initial implementation of standard Q-learning.

from __future__ import annotations

import random

import numpy as np

from src.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Standard tabular Q-learning with epsilon-greedy exploration."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(n_states, n_actions)
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        target = reward + self.gamma * best_next
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
