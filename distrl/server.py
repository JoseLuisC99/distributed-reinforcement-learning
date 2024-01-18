import torch
from torch import nn
import torch.distributed as dist
from typing import Optional
from itertools import count
from functools import reduce
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt


logger = logging.getLogger("Server")
logger.setLevel(logging.INFO)


class ParameterServer:
    def __init__(self, world_size: int, policy: nn.Module, lr: float = 0.001, max_episodes: Optional[int] = None):
        self.policy = policy.to(torch.device("cpu"))
        self.running_reward = 0
        self.__world_size = world_size
        self.__workers = world_size - 1
        self.__optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.__max_episodes = max_episodes

        self.__gradients = []
        self.__worker_buffer = []
        for param in self.policy.parameters():
            worker_return = []
            for _ in range(self.__world_size):
                worker_return.append(torch.empty(param.size()))
            self.__worker_buffer.append(worker_return)
            self.__gradients.append(torch.empty(param.size()))

    def _broadcast_parameters(self):
        for param in self.policy.parameters():
            dist.broadcast(param.detach(), src=0)

    def _receive_gradients(self):
        for idx, param in enumerate(self.policy.parameters()):
            dummy_grad = torch.empty(param.size())
            dist.gather(dummy_grad, self.__worker_buffer[idx], dst=0)
            self.__gradients[idx] = reduce(lambda x, y: x + y, self.__worker_buffer[idx][1:])

    def _receive_rewards(self):
        rewards = torch.tensor(float('inf'))
        dist.reduce(rewards, dst=0, op=dist.ReduceOp.MIN)
        return rewards

    def _update(self):
        for idx, param in enumerate(self.policy.parameters()):
            param.grad = self.__gradients[idx]
        self.__optimizer.step()

    def run(self):
        logger.info("Server waiting for workers.")
        iterator = range(self.__max_episodes) if self.__max_episodes is not None else count(1)
        running_reward_history = []
        for episode in tqdm(iterator):
            logger.info(f"Episode {episode}.")
            self.policy.train()
            self._broadcast_parameters()
            self._receive_gradients()
            self._update()
            min_reward = self._receive_rewards().item()
            self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward
            running_reward_history.append(min_reward)
        print(f"Global running reward: {self.running_reward:.3f}")

        return running_reward_history
