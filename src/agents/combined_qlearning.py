# Independently designed and implemented combined memory agent.

from __future__ import annotations

import random

import numpy as np

from src.agents.qlearning import QLearningAgent
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory


class CombinedAgent(QLearningAgent):
    """Q-learning with both short-term and long-term memory."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        memory_size: int = 10,
        repetition_penalty: float = 0.05,
        memory_strength: float = 0.10,
        memory_decay: float = 0.0,
    ) -> None:
        super().__init__(n_states, n_actions, alpha=alpha, gamma=gamma)
        self.short_term = ShortTermMemory(memory_size, repetition_penalty)
        self.long_term = LongTermMemory(
            n_states,
            n_actions,
            memory_strength=memory_strength,
            decay=memory_decay,
        )
        self.episode_repeats = 0
        self.episode_steps = 0

    def begin_episode(self) -> None:
        self.short_term.reset_episode()
        self.long_term.reset_episode()
        self.episode_repeats = 0
        self.episode_steps = 0

    def end_episode(self, success: bool) -> None:
        if success:
            self.long_term.store_successful_trajectory()

    def select_action(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        scores = []
        for action in range(self.n_actions):
            score = self.long_term.action_score(self.Q[state, action], state, action)
            if state in self.short_term.recent_states:
                score -= self.short_term.repetition_penalty
            scores.append(score)
        return int(np.argmax(scores))

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
        self.long_term.record_step(state, action)
        return self.short_term.adjust_reward(reward, is_repeat)

    def greedy_action(self, state: int) -> int:
        scores = []
        for action in range(self.n_actions):
            score = self.long_term.action_score(self.Q[state, action], state, action)
            scores.append(score)
        return int(np.argmax(scores))

    @property
    def repeated_state_rate(self) -> float:
        return self.short_term.repeated_state_rate()
