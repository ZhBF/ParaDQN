import numpy as np
from typing import Tuple


class ReplayBuffer:
    """High-efficiency replay buffer for parametrized actions.

    Stores tuples (s, a_idx, a_param, r, s_, done) in pre-allocated arrays.
    """

    def __init__(self, capacity: int, state_dim: int, param_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.param_dim = int(param_dim)
        self.size = 0
        self.index = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions_idx = np.zeros(capacity, dtype=np.int64)
        self.actions_param = np.zeros((capacity, param_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state: np.ndarray, action_idx: int, action_param: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add transition to buffer."""

        state = np.asarray(state, dtype=np.float32)
        action_param = np.asarray(action_param, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        self.states[self.index] = state
        self.actions_idx[self.index] = int(action_idx)
        self.actions_param[self.index] = action_param
        self.rewards[self.index] = float(reward)
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch uniformly."""

        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions_idx[indices],
            self.actions_param[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def get_memory_stats(self) -> dict:
        """Return memory usage statistics."""
        array_bytes = self.states.nbytes + self.actions_idx.nbytes + self.actions_param.nbytes + self.rewards.nbytes + self.next_states.nbytes + self.dones.nbytes
        return {
            "allocated_mb": array_bytes / (1024**2),
            "used_mb": (array_bytes * self.size / self.capacity) / (1024**2),
            "capacity": self.capacity,
            "size": self.size,
            "fill_ratio": self.size / self.capacity,
        }
