import threading
from typing import Tuple, SupportsFloat

import numpy as np
import torch


class MemoryReplay:
    def __init__(self, n: int, state_dim: Tuple[int, ...]):
        self._n = n
        self._next_index = 0

        # TODO: add dtype option
        self._states = torch.empty((n, *state_dim))
        self._actions = torch.empty(n)
        self._rewards = torch.empty(n)
        self._next_states = torch.empty((n, *state_dim))
        self._terminals = torch.empty(n)
        self.__lock = threading.Lock()

    def push(self,
             state: torch.Tensor,
             action: SupportsFloat,
             reward: SupportsFloat,
             next_state: torch.Tensor,
             is_terminal: bool) -> None:
        with self.__lock:
            idx = self._next_index
            self._next_index = (self._next_index + 1) % self._n
            self._states[idx] = state
            self._actions[idx] = float(action)
            self._rewards[idx] = float(reward)
            self._next_states[idx] = next_state
            self._terminals[idx] = int(is_terminal)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.__lock:
            # TODO: add preprocessing step (although it is better to use gym.Wrapper)
            indices = np.random.choice(range(self._n), batch_size)
            return (
                self._states[indices].type(torch.float32),
                self._actions[indices],
                self._rewards[indices],
                self._next_states[indices].type(torch.float32),
                self._terminals[indices]
            )

    def __len__(self) -> int:
        return self._n


class LinearAnnealer:
    def __init__(self, start_value, end_value, total_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> float:
        if self.current_step >= self.total_steps:
            return self.end_value
        else:
            value = self.start_value - (self.start_value - self.end_value) * (self.current_step / self.total_steps)
            self.current_step += 1
            return value

    def reset(self) -> None:
        self.current_step = 0
