# P-DQN: Parametrized Deep Q-Networks

This is an implementation of P-DQN, a reinforcement learning algorithm designed to handle environments with discrete-continuous hybrid action spaces.

## Quick Start

First, clone this repository and install the required packages.

```bash
git clone https://github.com/ZhBF/ParaDQN.git
cd ParaDQN
pip install -r requirements.txt
cd ..
```

Then, install the `gym-hybrid` environment.

```bash
git clone https://github.com/thomashirtz/gym-hybrid.git
cd gym-hybrid
pip install .
cd ..
```

Finally, run the training script.
```bash
cd ParaDQN
python main.py
```

## To Do Tasks
[x] Edit README with Quick Start instructions

[x] Add requirements.txt file

[x] Add command line arguments for training script

[x] Consider bounding and shifting action parameters

[ ] Implement more efficient storage in Replay Buffer

## Reference 
[Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space](https://arxiv.org/abs/1810.06394)
