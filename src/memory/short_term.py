"""Short-term memory for discouraging repeated state visits."""

from __future__ import annotations

from collections import deque


class ShortTermMemory:
    """Tracks recently visited states within the current episode."""

    def __init__(self, memory_size: int = 10, repetition_penalty: float = 0.05) -> None:
        self.memory_size = memory_size
        self.repetition_penalty = repetition_penalty
        self.recent_states: deque[int] = deque(maxlen=memory_size)
        self.repeated_transitions = 0
        self.total_transitions = 0

    def reset_episode(self) -> None:
        """Clear short-term memory at the start of a new episode."""
        self.recent_states.clear()

    def observe_transition(self, state: int) -> bool:
        """Record a transition and return whether the state was recently visited."""
        self.total_transitions += 1
        is_repeat = state in self.recent_states
        if is_repeat:
            self.repeated_transitions += 1
        self.recent_states.append(state)
        return is_repeat

    def adjust_reward(self, reward: float, is_repeat: bool) -> float:
        """Apply a repetition penalty when revisiting a recent state."""
        if is_repeat:
            return reward - self.repetition_penalty
        return reward

    def repeated_state_rate(self) -> float:
        """Fraction of transitions that revisited a recent state."""
        if self.total_transitions == 0:
            return 0.0
        return self.repeated_transitions / self.total_transitions

    def episode_repeated_state_rate(self, episode_repeats: int, episode_steps: int) -> float:
        """Episode-level repeated-state rate."""
        if episode_steps == 0:
            return 0.0
        return episode_repeats / episode_steps
