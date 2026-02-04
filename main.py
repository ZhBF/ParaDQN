from datetime import datetime
from environments import make_env
import torch
from replay_buffer import ReplayBuffer
from agent import ParaDQNAgent
import os
from torch.utils.tensorboard import SummaryWriter
from train import train
import argparse
import numpy as np
from vec_env import SubprocVecEnv


def main(args: argparse.Namespace):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("Using CPU")

    # environment (single env for specs)
    spec_env = make_env(args.env)
    if args.num_envs > 1:
        env = SubprocVecEnv(args.env, args.num_envs, seed=args.seed)
        eval_env = SubprocVecEnv(args.env, args.num_envs, seed=args.seed + 10000)
    else:
        env = make_env(args.env)
        eval_env = make_env(args.env)

    # buffer
    buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        state_dim=spec_env.state_dim,
        param_dim=spec_env.param_dim_total,
    )

    # agent
    agent = ParaDQNAgent(
        observation_space=spec_env.observation_space,
        action_space=spec_env.action_space,
        device=device,
        gamma=args.gamma,
        lr_q=args.lr_q,
        lr_actor=args.lr_actor,
        tau_q=args.tau_q,
        tau_actor=args.tau_actor,
        q_hidden_sizes=args.q_hidden_sizes,
        actor_hidden_sizes=args.actor_hidden_sizes,
    )

    # logging
    run_dir = os.path.join(
        os.path.dirname(__file__),
        "runs",
        f"run_{args.env}_{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(run_dir)
    print("Run dir:", run_dir)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir)

    # training
    train(
        env=env,
        agent=agent,
        buffer=buffer,
        writer=writer,
        episodes=args.train_episodes,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume_from,
        eval_env=eval_env,
    )

    writer.close()
    if hasattr(env, "close"):
        env.close()
    if hasattr(eval_env, "close"):
        eval_env.close()
    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # general settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu or cuda).")
    
    # environment settings
    parser.add_argument("--env", type=str, default="moving", help="Environment to use (moving, sliding).")
    
    # agent settings
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr_q", type=float, default=1e-3, help="Learning rate for Q network.")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="Learning rate for actor/param network.")
    parser.add_argument("--tau_q", type=float, default=0.005, help="Soft update factor for Q target network.")
    parser.add_argument("--tau_actor", type=float, default=0.005, help="Soft update factor for actor target network.")
    parser.add_argument("--q_hidden_sizes", type=int, nargs="+", default=[256, 256], help="Hidden layer sizes for Q network.")
    parser.add_argument("--actor_hidden_sizes", type=int, nargs="+", default=[256, 256], help="Hidden layer sizes for actor/param network.")

    # training settings
    parser.add_argument("--replay_capacity", type=int, default=20000, help="Replay buffer capacity.")
    parser.add_argument("--train_episodes", type=int, default=int(5e5), help="Number of training episodes.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--train_freq", type=int, default=1, help="Training frequency (in steps).")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--eval_interval", type=int, default=20, help="Evaluation interval (in episodes).")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (in episodes).")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting value of epsilon for epsilon-greedy.")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final value of epsilon for epsilon-greedy.")
    parser.add_argument("--epsilon_decay_steps", type=int, default=5000, help="Number of steps to decay epsilon.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume checkpoint.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    
    args = parser.parse_args()
    main(args)
