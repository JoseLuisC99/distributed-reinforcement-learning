import torch
from torch import nn
import torch.nn.functional as F
from distrl.a3c.node import Coordinator, Worker
from argparse import ArgumentParser
import gymnasium as gym
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 128)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x


class Actor(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.linear = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        x = self.linear(x)
        return F.softmax(x, dim=1)


class Critic(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.linear = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return self.linear(x)


def get_args():
    parser = ArgumentParser(description="A3C algorithm")
    parser.add_argument("--type", choices=["coordinator", "worker"], help="")
    parser.add_argument("--address", type=str, default="localhost", help="")
    parser.add_argument("--port", type=int, default=3000, help="")
    parser.add_argument("--gamma", type=float, default=0.9, help="")
    parser.add_argument("--bootstrap-steps", type=int, default=20, help="")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="")
    parser.add_argument("--update-freq", type=int, default=5, help="")
    parser.add_argument("--n-steps", type=int, help="")
    parser.add_argument("--version-tol", type=int, default=5, help="")
    parser.add_argument("--checkpoint-freq", type=int, required=False, help="")
    parser.add_argument("--max-workers", type=int, default=10, help="")
    parser.add_argument("--log", type=str, default="INFO", help="")
    parser.add_argument("--output", type=str, required=False, help="")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.log is not None:
        os.makedirs("logs", exist_ok=True)
        log_level = getattr(logging, args.log.upper(), None)
        if not isinstance(log_level, int):
            raise ValueError(f"Invalid log level: {args.log}")
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
            filename=os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d %H:%M:00')}.log"),
            encoding="utf-8",
            level=log_level
        )

    backbone = Backbone()
    actor = Actor(backbone)
    critic = Critic(backbone)

    if args.type == "coordinator":
        coordinator = Coordinator(
            actor, critic,
            args.learning_rate,
            args.update_freq,
            args.n_steps,
            args.bootstrap_steps,
            args.version_tol,
            args.checkpoint_freq,
            args.output
        )
        coordinator.start(args.max_workers, args.port)
    else:
        env = gym.make("CartPole-v1")
        worker = Worker(args.address, args.port, actor, critic, env)
        worker.run(args.gamma, args.bootstrap_steps)
