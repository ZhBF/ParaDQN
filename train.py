"""Train ParaDQN on a demo parametrized environment.

This script provides:
 - SimpleParamEnv: a tiny toy environment with parametrized actions for demo/training
 - train() function: runs episodes, stores transitions, samples from replay and updates agent
 - evaluate() function: runs greedy episodes to report average return

Adaptation notes:
 - Replace SimpleParamEnv with any environment that exposes:
     obs = env.reset()
     next_obs, reward, done, info = env.step((action_idx, action_param))
"""

import os
import time
import shutil
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import ParaDQNAgent
from replay_buffer import ReplayBuffer


class TestEnv:
    """A smaller deterministic test environment.

    - state_dim small (default 4)
    - each discrete action has a simple goal vector; action_param linearly shifts the state toward the goal
    - deterministic matrix W pre-generated and small noise
    """

    def __init__(self, state_dim: int = 4, num_actions: int = 2, param_dim_list: List[int] = [1, 1], max_steps: int = 20):
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


def evaluate(env, agent: ParaDQNAgent, episodes: int = 5):
    returns = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a_idx, a_param = agent.select_action(s, epsilon=0.0)
            s, r, done, _ = env.step((a_idx, a_param))
            total += r
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def train(
    env,
    agent: ParaDQNAgent,
    buffer: ReplayBuffer,
    episodes: int = 500,
    batch_size: int = 64,
    train_freq: int = 1,
    max_steps_per_episode: int = 50,
    eval_interval: int = 20,
    writer: Optional[SummaryWriter] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 50,
    resume_from: Optional[str] = None,
):
    total_steps = 0
    rewards_log = []
    start_episode = 1

    if resume_from is None and checkpoint_dir is not None:
        latest = os.path.join(checkpoint_dir, 'latest.pth')
        if os.path.exists(latest):
            resume_from = latest
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        ck = load_checkpoint(resume_from, agent)
        if ck is not None:
            total_steps = ck.get('total_steps', 0)
            start_episode = ck.get('episode', 1) + 1
            print(f"Resuming from episode {start_episode}, total_steps={total_steps}")
    
    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps_per_episode):
            eps = max(0.05, 1.0 - total_steps / 5000.0)
            a_idx, a_param = agent.select_action(s, epsilon=eps)
            s_, r, done, _ = env.step((a_idx, a_param))
            buffer.push(s.astype(np.float32), a_idx, np.pad(a_param.astype(np.float32), (0, env.param_dim_total - a_param.size)), float(r), s_.astype(np.float32), bool(done))
            s = s_
            ep_reward += r
            total_steps += 1

            if buffer.can_sample(batch_size) and total_steps % train_freq == 0:
                batch = buffer.sample(batch_size)
                info = agent.train_step(batch)
                if writer is not None and isinstance(info, dict):
                    if 'q_loss' in info:
                        writer.add_scalar('loss/q_loss', info['q_loss'], total_steps)
                    if 'actor_loss' in info:
                        writer.add_scalar('loss/actor_loss', info['actor_loss'], total_steps)
                    writer.add_scalar('train/epsilon', eps, total_steps)
                    writer.add_scalar('train/replay_size', len(buffer), total_steps)

            if done:
                break

        rewards_log.append(ep_reward)

        if ep % eval_interval == 0:
            mean_r, std_r = evaluate(env, agent, episodes=5)
            print(f"Ep {ep}/{episodes}  total_steps={total_steps}  recent_reward={np.mean(rewards_log[-eval_interval:]):.3f}  eval_mean={mean_r:.3f} +/- {std_r:.3f}")
            if writer:
                writer.add_scalar('eval/mean_return', mean_r, ep)
                writer.add_scalar('eval/std_return', std_r, ep)

        if writer:
            writer.add_scalar('episode/reward', ep_reward, ep)

        if checkpoint_dir and ep % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f'ck_ep{ep}.pth')
            save_checkpoint(path, agent, episode=ep, total_steps=total_steps)
            latest = os.path.join(checkpoint_dir, 'latest.pth')
            shutil.copy(path, latest)

    return rewards_log


def main():
    state_dim = 4
    param_dim_list = [1, 1]
    num_actions = len(param_dim_list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = TestEnv(state_dim=state_dim, num_actions=num_actions, param_dim_list=param_dim_list, max_steps=20)
    agent = ParaDQNAgent(state_dim=state_dim, actions_num=num_actions, actions_param_dim=param_dim_list, device=device)
    buffer = ReplayBuffer(capacity=20000, state_dim=state_dim, param_dim=sum(param_dim_list))

    run_dir = os.path.join(os.path.dirname(__file__), 'runs', f'run_{int(time.time())}')
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'logs'))
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    t0 = time.time()
    train(env, agent, buffer, episodes=500, batch_size=64, eval_interval=20, writer=writer, checkpoint_dir=checkpoint_dir, save_every=20)
    print("Training finished in", time.time() - t0)
    writer.close()


def save_checkpoint(path: str, agent: ParaDQNAgent, episode: int = 0, total_steps: int = 0):
    state = {
        'episode': int(episode),
        'total_steps': int(total_steps),
        'q_state_dict': agent.q_net.state_dict(),
        'actor_state_dict': agent.actor.state_dict(),
        'q_optimizer': agent.q_optimizer.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path: str, agent: ParaDQNAgent):
    try:
        ck = torch.load(path, map_location='cpu')
        if 'q_state_dict' in ck:
            agent.q_net.load_state_dict(ck['q_state_dict'])
        if 'actor_state_dict' in ck:
            agent.actor.load_state_dict(ck['actor_state_dict'])
        if 'q_optimizer' in ck and getattr(agent, 'q_optimizer', None) is not None and ck['q_optimizer'] is not None:
            try:
                agent.q_optimizer.load_state_dict(ck['q_optimizer'])
            except Exception:
                pass
        if 'actor_optimizer' in ck and getattr(agent, 'actor_optimizer', None) is not None and ck['actor_optimizer'] is not None:
            try:
                agent.actor_optimizer.load_state_dict(ck['actor_optimizer'])
            except Exception:
                pass
        return ck
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
        return None


if __name__ == '__main__':
    main()
