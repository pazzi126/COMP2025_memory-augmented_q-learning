# Independently designed and implemented short-term memory mechanism.

from __future__ import annotations

import random

import numpy as np

from src.agents.qlearning import QLearningAgent
from src.memory.short_term import ShortTermMemory


class STMAgent(QLearningAgent):
    """Q-learning augmented with short-term repetition penalties."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        memory_size: int = 10,
        repetition_penalty: float = 0.05,
    ) -> None:
        super().__init__(n_states, n_actions, alpha=alpha, gamma=gamma)
        self.short_term = ShortTermMemory(memory_size, repetition_penalty)
        self.episode_repeats = 0
        self.episode_steps = 0

    def begin_episode(self) -> None:
        self.short_term.reset_episode()
        self.episode_repeats = 0
        self.episode_steps = 0

    def select_action(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def on_step(
        self,
        state: int,
        action: int,
        next_state: int,
        reward: float,
    ) -> float:
        self.episode_steps += 1
        is_repeat = self.short_term.observe_transition(next_state)
        if is_repeat:
            self.episode_repeats += 1
        return self.short_term.adjust_reward(reward, is_repeat)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        super().update(state, action, reward, next_state, done)

    @property
    def repeated_state_rate(self) -> float:
        return self.short_term.repeated_state_rate()
