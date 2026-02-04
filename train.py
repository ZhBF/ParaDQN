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
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import ParaDQNAgent
from replay_buffer import ReplayBuffer


def save_checkpoint(path: str, agent: ParaDQNAgent, episode: int = 0, total_steps: int = 0):
    state = {
        "episode": int(episode),
        "total_steps": int(total_steps),
        "q_state_dict": agent.q_net.state_dict(),
        "actor_state_dict": agent.actor.state_dict(),
        "q_optimizer": agent.q_optimizer.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path: str, agent: ParaDQNAgent):
    try:
        ck = torch.load(path, map_location=agent.device)
        if "q_state_dict" in ck:
            agent.q_net.load_state_dict(ck["q_state_dict"])
        if "actor_state_dict" in ck:
            agent.actor.load_state_dict(ck["actor_state_dict"])
        if "q_optimizer" in ck and getattr(agent, "q_optimizer", None) is not None and ck["q_optimizer"] is not None:
            try:
                agent.q_optimizer.load_state_dict(ck["q_optimizer"])
            except Exception:
                pass
        if "actor_optimizer" in ck and getattr(agent, "actor_optimizer", None) is not None and ck["actor_optimizer"] is not None:
            try:
                agent.actor_optimizer.load_state_dict(ck["actor_optimizer"])
            except Exception:
                pass
        return ck
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
        return None


def evaluate(env, agent: ParaDQNAgent, episodes: int):
    num_envs = getattr(env, "num_envs", 1)
    returns = []

    if num_envs == 1:
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

    states = env.reset()
    ep_returns = [0.0 for _ in range(num_envs)]
    while len(returns) < episodes:
        actions = [agent.select_action(s, epsilon=0.0) for s in states]
        next_states, rewards, dones, _ = env.step(actions)
        for i in range(num_envs):
            ep_returns[i] += rewards[i]
            if dones[i]:
                returns.append(ep_returns[i])
                ep_returns[i] = 0.0
                if len(returns) >= episodes:
                    break
                next_states[i] = env.reset_at(i)
        states = next_states

    return float(np.mean(returns)), float(np.std(returns))


def train(
    env,
    agent: ParaDQNAgent,
    buffer: ReplayBuffer,
    writer: SummaryWriter,
    episodes: int,
    batch_size: int,
    train_freq: int,
    eval_episodes: int,
    eval_interval: int,
    save_interval: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    checkpoint_dir: str,
    resume_from: Optional[str] = None,
    eval_env=None,
):
    total_steps = 0
    rewards_log = []
    start_episode = 1

    # resume from checkpoint if provided
    if resume_from is not None and os.path.exists(resume_from):
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "latest.pth"
        if resume_path.exists():
            print(f"Loading checkpoint from {str(resume_path)}")
            ck = load_checkpoint(str(resume_path), agent)
            if ck is not None:
                total_steps = ck.get("total_steps", 0)
                start_episode = ck.get("episode", 1) + 1
                print(f"Resuming from episode {start_episode}, total_steps={total_steps}")

    num_envs = getattr(env, "num_envs", 1)
    eval_env = eval_env or env

    # main training loop
    if num_envs == 1:
        for ep in range(start_episode, episodes + 1):
            s = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                eps = max(epsilon_end, epsilon_start - total_steps / epsilon_decay_steps)
                a_idx, a_param = agent.select_action(s, epsilon=eps)
                s_, r, done, _ = env.step((a_idx, a_param))
                buffer.push(
                    s.astype(np.float32),
                    a_idx,
                    a_param.astype(np.float32),
                    float(r),
                    s_.astype(np.float32),
                    bool(done),
                )
                s = s_
                ep_reward += r
                total_steps += 1

                if buffer.can_sample(batch_size) and total_steps % train_freq == 0:
                    batch = buffer.sample(batch_size)
                    info = agent.train_step(batch)
                    if writer is not None and isinstance(info, dict):
                        if "q_loss" in info:
                            writer.add_scalar("loss/q_loss", info["q_loss"], total_steps)
                        if "actor_loss" in info:
                            writer.add_scalar("loss/actor_loss", info["actor_loss"], total_steps)
                        writer.add_scalar("train/epsilon", eps, total_steps)
                        writer.add_scalar("train/replay_size", len(buffer), total_steps)

                if done:
                    break

            rewards_log.append(ep_reward)

            if ep % eval_interval == 0:
                mean_r, std_r = evaluate(eval_env, agent, episodes=eval_episodes)
                print(
                    f"Ep {ep}/{episodes}  total_steps={total_steps}  "
                    f"recent_reward={np.mean(rewards_log[-eval_interval:]):.3f}  "
                    f"eval_mean={mean_r:.3f} +/- {std_r:.3f}"
                )
                if writer:
                    writer.add_scalar("eval/mean_return", mean_r, ep)
                    writer.add_scalar("eval/std_return", std_r, ep)

            if writer:
                writer.add_scalar("episode/reward", ep_reward, ep)

            if checkpoint_dir and ep % save_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"ck_ep{ep}.pth")
                save_checkpoint(path, agent, episode=ep, total_steps=total_steps)
                latest = os.path.join(checkpoint_dir, "latest.pth")
                shutil.copy(path, latest)
                print(f"Ep {ep}/{episodes}  Saved checkpoint to {path}.")

        return rewards_log

    # multi-env training
    completed_episodes = start_episode - 1
    states = env.reset()
    ep_rewards = [0.0 for _ in range(num_envs)]

    while completed_episodes < episodes:
        eps = max(epsilon_end, epsilon_start - total_steps / epsilon_decay_steps)
        actions = [agent.select_action(s, epsilon=eps) for s in states]
        next_states, rewards, dones, _ = env.step(actions)

        for i in range(num_envs):
            s = states[i]
            a_idx, a_param = actions[i]
            s_ = next_states[i]
            r = rewards[i]
            d = dones[i]
            buffer.push(
                s.astype(np.float32),
                a_idx,
                a_param.astype(np.float32),
                float(r),
                s_.astype(np.float32),
                bool(d),
            )
            ep_rewards[i] += r

            if d:
                completed_episodes += 1
                rewards_log.append(ep_rewards[i])
                if writer:
                    writer.add_scalar("episode/reward", ep_rewards[i], completed_episodes)
                ep_rewards[i] = 0.0
                if completed_episodes < episodes:
                    next_states[i] = env.reset_at(i)

                if completed_episodes % eval_interval == 0:
                    mean_r, std_r = evaluate(eval_env, agent, episodes=eval_episodes)
                    print(
                        f"Ep {completed_episodes}/{episodes}  total_steps={total_steps}  "
                        f"recent_reward={np.mean(rewards_log[-eval_interval:]):.3f}  "
                        f"eval_mean={mean_r:.3f} +/- {std_r:.3f}"
                    )
                    if writer:
                        writer.add_scalar("eval/mean_return", mean_r, completed_episodes)
                        writer.add_scalar("eval/std_return", std_r, completed_episodes)

                if checkpoint_dir and completed_episodes % save_interval == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    path = os.path.join(checkpoint_dir, f"ck_ep{completed_episodes}.pth")
                    save_checkpoint(path, agent, episode=completed_episodes, total_steps=total_steps)
                    latest = os.path.join(checkpoint_dir, "latest.pth")
                    shutil.copy(path, latest)
                    print(f"Ep {completed_episodes}/{episodes}  Saved checkpoint to {path}.")

        total_steps += num_envs

        if buffer.can_sample(batch_size) and total_steps % train_freq == 0:
            batch = buffer.sample(batch_size)
            info = agent.train_step(batch)
            if writer is not None and isinstance(info, dict):
                if "q_loss" in info:
                    writer.add_scalar("loss/q_loss", info["q_loss"], total_steps)
                if "actor_loss" in info:
                    writer.add_scalar("loss/actor_loss", info["actor_loss"], total_steps)
                writer.add_scalar("train/epsilon", eps, total_steps)
                writer.add_scalar("train/replay_size", len(buffer), total_steps)

        states = next_states

    return rewards_log
