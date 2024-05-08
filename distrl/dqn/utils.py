import threading
from typing import Tuple, SupportsFloat
from torch.distributed import rpc

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class MemoryReplay:
    def __init__(self,
                 n: int,
                 state_dim: Tuple[int, ...],
                 max_steps: int, *,
                 obs_dtype: torch.dtype = torch.float,
                 action_dtype: torch.dtype = torch.int8,
                 rewards_dtype: torch.dtype = torch.float):
        self._n = n
        self._next_index = 0
        self.__n_steps = 0
        self._max_steps = max_steps

        self._states = torch.empty((n, *state_dim), dtype=obs_dtype)
        self._actions = torch.empty(n, dtype=action_dtype)
        self._rewards = torch.empty(n, dtype=rewards_dtype)
        self._next_states = torch.empty((n, *state_dim), dtype=obs_dtype)
        self._terminals = torch.empty(n, dtype=torch.int8)
        self.__lock = threading.Lock()
        self.__listeners = []

    def subscribe(self, rref: rpc.RRef):
        logger.info(f"adding new listener to memory buffer: {rref}")
        self.__listeners.append(rref)

    def push(self,
             state: torch.Tensor,
             action: SupportsFloat,
             reward: SupportsFloat,
             next_state: torch.Tensor,
             is_terminal: bool) -> None:
        with self.__lock:
            idx = self._next_index
            self._next_index = (self._next_index + 1) % self._n
            self.__n_steps += 1

            self._states[idx] = state
            self._actions[idx] = float(action)
            self._rewards[idx] = float(reward)
            self._next_states[idx] = next_state
            self._terminals[idx] = int(is_terminal)

            if self.__n_steps >= self._max_steps:
                logger.info(f"disconnecting memory after {self.__n_steps} samples")
                for rref in self.__listeners:
                    rref.rpc_sync().stop()

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(range(self._n), batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._terminals[indices]
        )

    def stats(self):
        logger.info(f"Memory samples: {self.__n_steps}")
        logger.info(f"Action stats: {self._actions.unique(return_counts=True)}")

    def __len__(self) -> int:
        return self._n


class LinearAnnealer:
    def __init__(self, start_value, end_value, total_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.current_step = 0
        self.current_value = start_value

    def step(self) -> float:
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.current_value = self.end_value
        else:
            self.current_value = (self.start_value - (self.start_value - self.end_value) *
                                  (self.current_step / self.total_steps))
        return self.current_value

    def reset(self) -> None:
        self.current_step = 0
