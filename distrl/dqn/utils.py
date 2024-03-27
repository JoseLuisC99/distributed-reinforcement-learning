import torch
import numpy as np
from typing import Tuple, SupportsFloat


class MemoryReplay:
    def __init__(self, n: int, state_dim: Tuple[int, ...], device: torch.device):
        self._n = n
        self._size = 0
        self._next_index = 0

        self._states = np.empty((n, *state_dim), dtype=np.float32)
        self._actions = np.empty(n, dtype=np.int32)
        self._rewards = np.empty(n, dtype=np.float32)
        self._next_states = np.empty((n, *state_dim), dtype=np.float32)
        self._terminals = np.empty(n, dtype=np.int32)
        self._device = device

    def push(self,
             state: np.ndarray,
             action: torch.Tensor,
             reward: SupportsFloat,
             next_state: np.ndarray,
             is_terminal: bool) -> None:
        self._states[self._next_index] = state
        self._actions[self._next_index] = action
        self._rewards[self._next_index] = reward
        self._next_states[self._next_index] = next_state
        self._terminals[self._next_index] = is_terminal

        self._next_index = (self._next_index + 1) % self._n
        self._size = min(self._size + 1, self._n)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(range(self._size), batch_size)
        return (
            torch.FloatTensor(self._states[indices]).to(self._device),
            torch.LongTensor(self._actions[indices]).to(self._device),
            torch.FloatTensor(self._rewards[indices]).to(self._device),
            torch.FloatTensor(self._next_states[indices]).to(self._device),
            torch.LongTensor(self._terminals[indices]).to(self._device)
        )

    def __len__(self) -> int:
        return self._size


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
