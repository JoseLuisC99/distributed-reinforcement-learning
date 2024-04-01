# Distributed Reinforcement Learning

`Distributed Reinforcement Learning (DistRL) is a Python project designed to implement distributed deep reinforcement 
learning in PyTorch with minimal dependencies. The primary goal is to create a flexible framework for experimenting with 
multi-objective and meta-learning in a distributed environment.

## Installation
**Requirements**: Python >= 3.11.6

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install DistRL:

```console
git clone https://github.com/JoseLuisC99/distributed-reinforcement-learning
cd distributed-reinforcement-learning
pip install -r requirements.txt
pip install -e .
```

## Usage
At the moment, only the next models are available:

- [x] Policy-Parameter Server
- [x] GORILA
- [ ] A3C
- [ ] IMPALA
- [ ] Ape-X
- [ ] R2D2
- [ ] SEED RL

### Policy-Parameter Server
This demo model only supports [Gymnasium](https://gymnasium.farama.org/) environments. modify the policy network in the 
file [launcher.py](https://github.com/JoseLuisC99/distributed-reinforcement-learning/blob/main/scripts/launcher.py) and 
then execute the next command:

This demo model currently supports [Gymnasium environments](https://gymnasium.farama.org/). To use it, modify the policy 
network in the file [launcher.py](https://github.com/JoseLuisC99/distributed-reinforcement-learning/blob/main/scripts/launcher.py) 
and then execute the following command:

```console
usage: launcher.py --workers WORKERS --master_port MASTER_PORT --env_name ENV_NAME --max_iters MAX_ITERS --max_episodes MAX_EPISODES --output_dir OUTPUT_DIR
```

Ensure that `MASTER_PORT` is a free port on your computer, and `ENV_NAME` is a valid environment ID.

## License

[MIT License](https://github.com/JoseLuisC99/distributed-reinforcement-learning/blob/main/LICENSE)