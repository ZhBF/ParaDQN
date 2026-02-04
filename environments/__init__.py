from environments.moving import MovingEnv
from environments.sliding import SlidingEnv


def make_env(env_name: str):
    if env_name == "sliding":
        return SlidingEnv()
    if env_name == "moving":
        return MovingEnv()
    raise ValueError(f"Unknown environment: {env_name}")
