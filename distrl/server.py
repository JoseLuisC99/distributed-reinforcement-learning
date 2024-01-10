import torch
from torch import nn
import torch.distributed as dist
from typing import Optional
from itertools import count
from functools import reduce
import logging
import time
from tqdm import tqdm


logger = logging.getLogger("Server")
logger.setLevel(logging.INFO)


class ParameterServer:
    def __init__(self, world_size: int, policy: nn.Module, lr: float = 0.001,
                 max_episodes: Optional[int] = None, timeout: Optional[float] = None):
        self.policy = policy.to(torch.device("cpu"))
        self.__world_size = world_size
        self.__workers = world_size - 1
        self.__timeout = timeout
        self.__optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.__max_episodes = max_episodes

        self.__gradients = []
        self.__gradients_buffer = []
        for param in self.policy.parameters():
            param_gradients = []
            for _ in range(self.__world_size):
                param_gradients.append(torch.empty(param.size()))
            self.__gradients_buffer.append(param_gradients)
            self.__gradients.append(torch.empty(param.size()))

    def _broadcast_parameters(self, episode: int):
        broadcast_time = time.time()
        for param in self.policy.parameters():
            dist.broadcast(param.detach(), src=0)
        broadcast_time = time.time() - broadcast_time
        logger.info(f"Broadcast time on episode {episode}: {broadcast_time:.4f}s")

    def _receive_gradients(self, episode: int):
        receive_time = time.time()
        for idx, param in enumerate(self.policy.parameters()):
            dummy_grad = torch.empty(param.size())
            dist.gather(dummy_grad, self.__gradients_buffer[idx], dst=0)
            self.__gradients[idx] = reduce(lambda x, y: x + y, self.__gradients_buffer[idx][1:])
        receive_time = time.time() - receive_time
        logger.info(f"Receiving time on episode {episode}: {receive_time:.4f}s")

    def _update(self, episode: int):
        update_time = time.time()
        self.__gradients = [grad / self.__workers for grad in self.__gradients]
        for idx, param in enumerate(self.policy.parameters()):
            param.grad = self.__gradients[idx]
        self.__optimizer.step()
        update_time = time.time() - update_time
        logger.info(f"Update model time on episode {episode}: {update_time:.4f}s")

    def run(self):
        logger.info("Server waiting for workers.")
        iterator = range(self.__max_episodes) if self.__max_episodes is not None else count(1)
        for episode in tqdm(iterator):
            logger.info(f"Episode {episode}.")
            self.policy.train()
            self._broadcast_parameters(episode)
            self._receive_gradients(episode)
            self._update(episode)
