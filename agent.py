import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from networks import QNetwork, ParamNetwork
from gym import spaces


class ParaDQNAgent:
    """ ParaDQN agent implementation.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Tuple,
                 device: str,
                 gamma: float,
                 lr_q: float,
                 lr_actor: float,
                 tau_q: float,
                 tau_actor: float):
        """ Initialize ParaDQNAgent.
        
        Inputs:
            - observation_space: observation space of the environment
            - action_space: action space of the environment (discrete + continuous)
            - device: device to run the networks on
            - lr_q: learning rate for Q network
            - lr_actor: learning rate for actor/param network
            - gamma: discount factor
            - tau: soft update factor for target networks
        """
        
        self.state_dim = observation_space.shape[0]
        self.actions_num = action_space[0].n
        self.param_dim = action_space[1].shape[0]
        self.param_space = action_space[1]
        self.device = torch.device(device)

        self.q_net = QNetwork(self.state_dim, self.actions_num, self.param_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        self.actor = ParamNetwork(self.state_dim, self.param_space).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.gamma = gamma
        self.tau_q = tau_q
        self.tau_actor = tau_actor

    @torch.no_grad()
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, np.ndarray]:
        """ Return discrete action index and continuous parameter vector for that action.

        Inputs:
            - state: (state_dim,) current state
            - epsilon: float, probability of choosing a random action for epsilon-greedy exploration
        Returns: 
            - action_idx: int
            - action_param: np.ndarray, parameter vector for that action
        """
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(0, self.actions_num)
            params = np.random.uniform(-1.0, 1.0, size=(self.param_dim,))
            a_param = params
            return int(a_idx), a_param

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, state_dim)
        params = self.actor(s)           # (1, param_dim)
        q_vals = self.q_net(s, params)   # (1, actions_num)
        q_vals = q_vals.cpu().numpy()[0]
        best_a_idx = int(q_vals.argmax())
        best_a_param = params.cpu().numpy()[0]
        return best_a_idx, best_a_param

    def train_step(self, batch, q_loss_coef: float = 1.0, actor_loss_coef: float = 1.0):
        """ Perform one training step from a batch.

        Inputs:
            - batch: (batch_size, (states, a_idx, a_params, rewards, next_states, dones))
            - q_loss_coef: coefficient for Q loss
            - actor_loss_coef: coefficient for actor loss
        Returns:
            - info: dict with 'q_loss' and 'actor_loss' values
        """
        states, a_idxs, params, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        a_idxs = torch.tensor(a_idxs, dtype=torch.int64, device=self.device)
        params = torch.tensor(params, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_params = self.actor_target(next_states)  # (batch, param_dim)
            q_next_all = self.q_target(next_states, next_params)  # (batch, actions_num)
            q_next_max, _ = q_next_all.max(dim=1)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next_max

        q_all = self.q_net(states, params)
        q_pred = q_all.gather(1, a_idxs.view(-1, 1)).squeeze(1)

        q_loss = nn.MSELoss()(q_pred, q_target)

        self.q_optimizer.zero_grad()
        (q_loss * q_loss_coef).backward()
        self.q_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_params = self.actor(states)  # (batch, param_dim)
        q_for_actions = self.q_net(states, actor_params)  # (batch, actions_num)
        q_val, a_chosen = q_for_actions.max(dim=1)
        actor_loss = -(q_val.mean())
        (actor_loss * actor_loss_coef).backward()
        self.actor_optimizer.step()

        self.soft_update(self.q_target, self.q_net, self.tau_q)
        self.soft_update(self.actor_target, self.actor, self.tau_actor)

        return {'q_loss': q_loss.item(), 'actor_loss': actor_loss.item()}

    def soft_update(self, target_net: nn.Module, net: nn.Module, tau: float):
        """ Soft update target network parameters.
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)
