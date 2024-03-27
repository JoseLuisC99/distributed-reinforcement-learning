import os
import copy
import torch
from torch import nn, optim
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from distrl.dqn.utils import MemoryReplay, LinearAnnealer
from typing import Any, Optional
from itertools import count
import random


class DeepQAgent:
    def __init__(self, q_network: nn.Module, env: gym.Wrapper, gamma: float, replay_memory_size: int,
                 replay_start: int, batch_size: int, train_frequency: int, update_frequency: int,
                 checkpoint_frequency: Optional[int] = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        self._batch_size = batch_size
        self._env = env
        self._train_frequency = train_frequency
        self._update_frequency = update_frequency
        self._checkpoint_frequency = checkpoint_frequency
        self._replay_start = replay_start
        self._gamma = gamma

        self._target_network = copy.deepcopy(q_network).to(device)
        self._q_network = q_network.to(device)
        self._memory_replay = MemoryReplay(replay_memory_size, self._env.observation_space.shape, device)
        self._device = device

        if self._checkpoint_frequency is not None:
            os.makedirs("checkpoints", exist_ok=True)

    def _epsilon_greedy(self, epsilon: float, state: torch.Tensor) -> Any:
        assert 0.0 <= epsilon <= 1.0
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self._q_network(state)
                action = torch.argmax(q_values, dim=1)
        else:
            action = torch.LongTensor([self._env.action_space.sample()])
        return action

    def train(self, n_frames: int, optimizer: optim.Optimizer, epsilon: LinearAnnealer) -> nn.Module:
        progress_bar = tqdm(range(n_frames))
        steps = 0
        for i in count():
            state, _ = self._env.reset()
            done = False

            while not done:
                action = self._epsilon_greedy(
                    epsilon.step(), torch.Tensor(np.array(state)).unsqueeze(0).to(self._device)
                )
                next_state, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                steps += 1
                progress_bar.update()
                self._memory_replay.push(state, action, min(max(float(reward), -1), 1), next_state, 1 * done)
                state = next_state

                if steps % self._train_frequency == 0:
                    if steps < self._replay_start:
                        continue
                    states, actions, rewards, next_states, terminals = self._memory_replay.sample(self._batch_size)

                    with torch.no_grad():
                        expected_q_values = torch.max(self._target_network(next_states), dim=1)[0]
                    # print(expected_q_values)
                    y = rewards + (1 - terminals) * self._gamma * expected_q_values
                    # print(torch.any(1 - terminals == 0))
                    # print(y.shape, self._q_network(states)[range(self._batch_size), actions].shape)
                    # print(self._q_network(states)[range(self._batch_size), actions])
                    loss = nn.SmoothL1Loss()(y, self._q_network(states)[range(self._batch_size), actions])

                    optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self._q_network.parameters(), 1.0)
                    for param in self._q_network.parameters():
                        param.grad.data.clamp_(-1, 1)
                    optimizer.step()

                if steps % self._update_frequency == 0:
                    # print(f"Updating model: qfunc_{steps}.pth")
                    self._target_network.load_state_dict(self._q_network.state_dict())

                if self._checkpoint_frequency is not None and steps % self._checkpoint_frequency == 0:
                    torch.save(self._target_network.state_dict(), os.path.join("checkpoints", f"deep_q_learning{steps}.pth"))

                if steps >= n_frames:
                    print(f"{i} episodes lapsed")
                    return self._target_network
