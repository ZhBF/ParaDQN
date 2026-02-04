# P-DQN: Parametrized Deep Q-Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of **Parametrized Deep Q-Networks (P-DQN)**, a reinforcement learning algorithm designed to handle environments with **discrete-continuous hybrid action spaces**.

## 🌟 Features

- **Hybrid Action Space Support**: Seamlessly handles both discrete and continuous action components
- **Modular Architecture**: Clean separation of agent, networks, and replay buffer
- **Flexible Training**: Configurable hyperparameters via command-line arguments
- **Tensorboard Integration**: Real-time monitoring of training metrics
- **Checkpoint System**: Automatic saving and loading of model checkpoints
- **Multiple Environments**: Support for various gym-hybrid environments (Moving, Sliding, etc.)

## 📋 Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gym
- Tensorboard

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ZhBF/ParaDQN.git
cd ParaDQN
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install PyTorch (visit https://pytorch.org for your specific configuration)
pip install torch torchvision torchaudio
```

### 3. Install Gym-Hybrid Environment

```bash
cd ..
git clone https://github.com/thomashirtz/gym-hybrid.git
cd gym-hybrid
pip install -e .
cd ../ParaDQN
```

### 4. Run Training

```bash
# Train with default parameters
python main.py

# Train with custom parameters
python main.py --env_name moving --episodes 2000 --lr 0.001
```

## 📊 Project Structure

```
ParaDQN/
├── agent.py              # P-DQN agent implementation
├── networks.py           # Neural network architectures
├── replay_buffer.py      # Experience replay buffer
├── train.py              # Training loop and logic
├── main.py               # Entry point with argument parsing
├── requirements.txt      # Python dependencies
├── environments/         # Custom environment wrappers
│   ├── moving.py
│   └── sliding.py
└── runs/                 # Training logs and checkpoints
```

## 🎮 Usage

### Training Arguments

```bash
python main.py [OPTIONS]

Options:
  --env_name          Environment name (default: 'moving')
  --episodes          Number of training episodes (default: 1600)
  --batch_size        Batch size for training (default: 128)
  --lr                Learning rate (default: 0.0001)
  --gamma             Discount factor (default: 0.99)
  --buffer_size       Replay buffer size (default: 100000)
  --tau               Target network update rate (default: 0.001)
```

### Monitoring Training

```bash
# View training progress with Tensorboard
tensorboard --logdir runs/
```

Then open your browser to `http://localhost:6006`

### Loading Checkpoints

Checkpoints are automatically saved in `runs/run_<env>_<id>_<timestamp>/checkpoints/`. To resume training or evaluate a trained model, load the checkpoint in your script.

## 📈 Results

The agent learns to solve hybrid action space tasks effectively. Training progress can be monitored through Tensorboard, showing metrics such as:
- Episode rewards
- Q-value estimates
- Loss curves
- Success rates

## 🔧 To Do

- [ ] Enable multi-processing for faster training

## 📝 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{xiong2018parametrized,
  title={Parametrized deep q-networks learning: Reinforcement learning with discrete-continuous hybrid action space},
  author={Xiong, Jiechao and Wang, Qing and Yang, Zhuoran and Sun, Peng and Han, Lei and Zheng, Yang and Fu, Haobo and Zhang, Tong and Liu, Ji and Liu, Han},
  journal={arXiv preprint arXiv:1810.06394},
  year={2018}
}
```

## 📚 References

- [Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space](https://arxiv.org/abs/1810.06394)
- [Gym-Hybrid Environment](https://github.com/thomashirtz/gym-hybrid)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ✉️ Contact

For questions or discussions, please open an issue on GitHub.
