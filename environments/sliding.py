import gym
import numpy as np
from typing import Tuple
import gym_hybrid


class SlidingEnv:
    def __init__(self):
        self.env = gym.make("Sliding-v0")
        self.state_dim = 10
        self.num_actions = 3
        self.param_dim_list = [1, 1, 0]
        self.param_dim_total = 2

    def reset(self):
        s = self.env.reset()
        s = np.array(s, dtype=np.float32)
        return s

    def step(self, action_tuple: Tuple[int, np.ndarray]):
        s, r, done, info = self.env.step(action_tuple)
        s = np.array(s, dtype=np.float32)
        r = float(r)
        return s, r, done, info
