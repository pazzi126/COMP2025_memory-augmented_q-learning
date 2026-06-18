"""Long-term memory for successful state-action trajectories."""

from __future__ import annotations

import numpy as np


class LongTermMemory:
    """Stores state-action pairs from successful episodes."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        memory_strength: float = 0.10,
        decay: float = 0.0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_strength = memory_strength
        self.decay = decay
        self.memory = np.zeros((n_states, n_actions), dtype=np.float64)
        self.current_trajectory: list[tuple[int, int]] = []

    def reset_episode(self) -> None:
        """Start tracking a new episode trajectory."""
        self.current_trajectory.clear()

    def record_step(self, state: int, action: int) -> None:
        """Record a state-action pair from the current episode."""
        self.current_trajectory.append((state, action))

    def store_successful_trajectory(self) -> None:
        """Increase memory values for actions in a successful trajectory."""
        if not self.current_trajectory:
            return

        if self.decay > 0:
            self.memory *= 1.0 - self.decay

        for state, action in self.current_trajectory:
            self.memory[state, action] += 1.0

        max_value = self.memory.max()
        if max_value > 0:
            self.memory /= max_value

    def get_bonus(self, state: int, action: int) -> float:
        """Return the normalized long-term memory bonus."""
        return float(self.memory[state, action])

    def action_score(self, q_value: float, state: int, action: int) -> float:
        """Combine Q-value with a long-term memory bonus."""
        return q_value + self.memory_strength * self.get_bonus(state, action)
