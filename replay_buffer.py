import random
import numpy as np
from collections import deque
from typing import Deque


class ReplayBuffer:
    """ Replay buffer for parametrized actions.

    Stores tuples (s, a_idx, a_param, r, s_, done)

    TODO: use more efficient storage
    """

    def __init__(self, capacity: int, state_dim: int, param_dim: int):
        self.capacity = int(capacity)
        self.buffer: Deque = deque(maxlen=self.capacity)
        self.state_dim = state_dim
        self.param_dim = param_dim

    def push(self, state: np.ndarray, action_idx: int, action_param: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((np.array(state, copy=True), int(action_idx), np.array(action_param, copy=True), 
                            float(reward), np.array(next_state, copy=True), bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, a_idx, a_params, rewards, next_states, dones = zip(*batch)
        return (np.vstack(states), np.array(a_idx, dtype=np.int64), np.vstack(a_params), np.array(rewards, dtype=np.float32), np.vstack(next_states), np.array(dones, dtype=np.float32))

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def __len__(self):
        return len(self.buffer)
