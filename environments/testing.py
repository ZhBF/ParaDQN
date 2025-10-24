# import gym
import numpy as np
from typing import Tuple, List


class TestEnv:
    """A smaller deterministic test environment.

    - state_dim small (default 4)
    - each discrete action has a simple goal vector; action_param linearly shifts the state toward the goal
    - deterministic matrix W pre-generated and small noise
    """

    def __init__(
        self,
        state_dim: int = 4,
        num_actions: int = 2,
        param_dim_list: List[int] = [1, 1],
        max_steps: int = 20,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.param_dim_list = param_dim_list
        self.param_dim_total = sum(param_dim_list)
        self.max_steps = max_steps
        rng = np.random.RandomState(0)
        self.goals = [rng.randn(state_dim) * 0.2 for _ in range(num_actions)]
        self.W_list = []
        for a in range(num_actions):
            size = param_dim_list[a]
            W = rng.randn(state_dim, size) if size > 0 else np.zeros((state_dim, 0))
            self.W_list.append(W)
        self.reset()

    def reset(self):
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.step_cnt = 0
        return self.state.copy()

    def step(self, action_tuple: Tuple[int, np.ndarray]):
        a_idx, a_param = action_tuple
        self.step_cnt += 1
        W = self.W_list[a_idx]
        if a_param.size == 0:
            param_effect = np.zeros(self.state_dim)
        else:
            param_effect = W.dot(a_param)
        goal = self.goals[a_idx]
        self.state = 0.8 * self.state + 0.2 * (goal + 0.1 * param_effect)
        dist = np.linalg.norm(self.state - goal)
        reward = -dist
        done = self.step_cnt >= self.max_steps
        return self.state.copy(), float(reward), done, {}
