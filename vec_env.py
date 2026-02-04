import multiprocessing as mp
import random
from typing import List, Tuple

import numpy as np

from environments import make_env


def _worker(remote, parent_remote, env_name: str, seed: int | None):
    parent_remote.close()
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:
            pass

    env = make_env(env_name)
    try:
        env.reset(seed=seed)
    except Exception:
        pass

    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            obs = env.reset()
            remote.send(obs)
        elif cmd == "step":
            action = data
            obs, reward, done, info = env.step(action)
            remote.send((obs, reward, done, info))
        elif cmd == "close":
            remote.close()
            break
        else:
            raise ValueError(f"Unknown cmd: {cmd}")


class SubprocVecEnv:
    def __init__(self, env_name: str, num_envs: int, seed: int | None = None):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.env_name = env_name
        self._closed = False

        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            env_seed = None if seed is None else int(seed) + i
            p = ctx.Process(target=_worker, args=(work_remote, remote, env_name, env_seed))
            p.daemon = True
            p.start()
            work_remote.close()
            self.processes.append(p)

    def reset(self) -> List[np.ndarray]:
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def reset_at(self, index: int) -> np.ndarray:
        remote = self.remotes[index]
        remote.send(("reset", None))
        return remote.recv()

    def step(self, actions: List[Tuple[int, np.ndarray]]):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    def close(self):
        if self._closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self._closed = True
