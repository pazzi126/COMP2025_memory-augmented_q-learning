# Independently designed and implemented long-term state-action memory.

from __future__ import annotations

import random

import numpy as np

from src.agents.qlearning import QLearningAgent
from src.memory.long_term import LongTermMemory


class LTMAgent(QLearningAgent):
    """Q-learning augmented with long-term successful trajectory memory."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        memory_strength: float = 0.10,
        memory_decay: float = 0.0,
    ) -> None:
        super().__init__(n_states, n_actions, alpha=alpha, gamma=gamma)
        self.long_term = LongTermMemory(
            n_states,
            n_actions,
            memory_strength=memory_strength,
            decay=memory_decay,
        )

    def begin_episode(self) -> None:
        self.long_term.reset_episode()

    def end_episode(self, success: bool) -> None:
        if success:
            self.long_term.store_successful_trajectory()

    def select_action(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        scores = [
            self.long_term.action_score(self.Q[state, action], state, action)
            for action in range(self.n_actions)
        ]
        return int(np.argmax(scores))

    def on_step(
        self,
        state: int,
        action: int,
        next_state: int,
        reward: float,
    ) -> float:
        self.long_term.record_step(state, action)
        return reward

    def greedy_action(self, state: int) -> int:
        scores = [
            self.long_term.action_score(self.Q[state, action], state, action)
            for action in range(self.n_actions)
        ]
        return int(np.argmax(scores))
