import torch
from torch import nn
import torch.distributed as dist
from typing import Optional
from itertools import count
from functools import reduce
from tqdm import tqdm


class ParameterServer:
    def __init__(self, world_size: int, policy: nn.Module, lr: float = 0.001, max_episodes: Optional[int] = None):
        self.policy = policy.to(torch.device("cpu"))
        self.running_reward = 0
        self.__optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.__max_episodes = max_episodes

        self.__gradients = []
        self.__worker_buffer = []
        for param in self.policy.parameters():
            worker_return = []
            for _ in range(world_size):
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

    def run(self, smoothing_factor: float = 0.9):
        iterator = range(self.__max_episodes) if self.__max_episodes is not None else count(1)
        running_reward_history = []
        for _ in tqdm(iterator):
            self.policy.train()
            self._broadcast_parameters()
            self._receive_gradients()
            self._update()
            min_reward = self._receive_rewards().item()
            self.running_reward = (1 - smoothing_factor) * min_reward + smoothing_factor * self.running_reward
            running_reward_history.append(self.running_reward)

        return running_reward_history
