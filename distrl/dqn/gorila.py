import copy
import os
import random
import threading
from itertools import count
from math import ceil
from typing import Any, Optional, Dict, Tuple, List, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributed import rpc

from distrl.dqn.utils import LinearAnnealer


# TODO: add logging option
class ParameterServer:
    def __init__(self,
                 q_network: nn.Module,
                 lr: float,
                 sync_frequency: int,
                 max_version_difference: Optional[int] = None,
                 checkpoint: bool = True):
        self._q_network = q_network
        self._target_network = copy.deepcopy(q_network)
        self.__version = 0
        self._optimizer = optim.AdamW(self._q_network.parameters(), lr=lr, amsgrad=True)
        self._max_version_difference = max_version_difference
        self._sync_frequency = sync_frequency
        self._checkpoint = checkpoint
        self.__listeners = []
        self.__lock = threading.Lock()

        if self._checkpoint:
            os.makedirs("checkpoints", exist_ok=True)

    def update(self, version: int, gradients: Dict[str, torch.Tensor]) -> bool:
        with self.__lock:
            if abs(self.__version - version) > self._max_version_difference:
                return False

            self.__version += 1
            self._optimizer.zero_grad()
            for name, param in self._q_network.named_parameters():
                if param.requires_grad:
                    param.grad = gradients[name]
            self._optimizer.step()

            if self.__version % self._sync_frequency == 0:
                self.sync_target_network()

        return True

    def parameters(self) -> Tuple[int, Dict[str, Any]]:
        with self.__lock:
            return self.__version, self._q_network.state_dict()

    def subscribe(self, rref: rpc.RRef):
        self.__listeners.append(rref)

    def sync_target_network(self) -> None:
        state_dict = self._q_network.state_dict()
        self._target_network.load_state_dict(state_dict)
        futures = []
        for rref in self.__listeners:
            futures.append(rref.rpc_async().sync(state_dict))
        for fut in futures:
            fut.wait()

        if self._checkpoint:
            torch.save(state_dict, os.path.join("checkpoints", f"checkpoint-{self.__version}.pth"))

    def result(self) -> nn.Module:
        return self._target_network


# TODO: add logging option
class Learner:
    def __init__(self,
                 q_network: nn.Module,
                 parameter_server: rpc.RRef,
                 memory_replay: rpc.RRef,
                 batch_size: int,
                 gamma: Optional[float] = 0.99,
                 device: torch.device = torch.device("cpu")):
        self._device = device
        self._q_network = q_network.to(self._device)
        self._target_network = copy.deepcopy(q_network).to(self._device)
        self._parameter_server = parameter_server
        self._memory_replay = memory_replay
        self._batch_size = batch_size
        self._gamma = gamma
        self.__version = 0
        self.__target_version = 0
        self.__stop_signal = False
        self.__lock = threading.Lock()

    def sync(self, state_dict: Dict[str, Any]):
        with self.__lock:
            self._target_network.load_state_dict(state_dict)

    def stop(self):
        self.__stop_signal = False

    def train(self, n_steps: int):
        step = 0
        while not self.__stop_signal:
            version, state_dict = self._parameter_server.rpc_sync().parameters()
            self._q_network.load_state_dict(state_dict)
            self.__version = version

            for param in self._q_network.parameters():
                param.grad = None

            states, actions, rewards, next_states, terminals = self._memory_replay.rpc_sync().sample(self._batch_size)
            states = states.to(self._device)
            actions = actions.type(torch.int64).to(self._device)
            rewards = rewards.to(self._device)
            next_states = next_states.to(self._device)
            terminals = terminals.to(self._device)

            with torch.no_grad():
                expected_q_values = torch.max(self._target_network(next_states), dim=1)[0]
            y = rewards + (1 - terminals) * self._gamma * expected_q_values
            loss = nn.SmoothL1Loss()(y, self._q_network(states)[range(self._batch_size), actions])

            loss.backward()
            for param in self._q_network.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

            gradients = {}
            for name, param in self._q_network.named_parameters():
                gradients[name] = param.grad.data.detach().cpu()
            if not self._parameter_server.rpc_sync().update(self.__version, gradients):
                pass
            step += 1
            if step == n_steps:
                break


# TODO: add logging option
class Actor:
    def __init__(self,
                 env: Union[gym.Wrapper, gym.Env],
                 memory_replay: rpc.RRef,
                 parameter_server: rpc.RRef,
                 q_network: nn.Module,
                 annealer: LinearAnnealer,
                 update_frequency: int,
                 device: torch.device = torch.device("cpu")):
        self._env = env
        self._memory_replay = memory_replay
        self._parameter_server = parameter_server
        self._annealer = annealer
        self._device = device
        self._update_frequency = update_frequency
        self._q_network = q_network.to(self._device)
        self.__stop_signal = False

        self.__n_samples = 0

    def update(self):
        _, state_dict = self._parameter_server.rpc_sync().parameters()
        self._q_network.load_state_dict(state_dict)

    def _epsilon_greedy(self, epsilon: float, state: torch.Tensor) -> Any:
        assert 0.0 <= epsilon <= 1.0
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self._q_network(state)
                action = torch.argmax(q_values, dim=1).detach().cpu().item()
        else:
            action = self._env.action_space.sample()
        return action

    def stop(self):
        self.__stop_signal = True

    def act(self, n_steps: Optional[int] = None):
        step = 0
        for _ in count():
            state, _ = self._env.reset()
            done = False

            while not done:
                self.__n_samples += 1
                if self.__stop_signal:
                    return
                if step % self._update_frequency == 0:
                    self.update()

                step += 1
                action = self._epsilon_greedy(
                    self._annealer.step(), torch.Tensor(np.array(state)).unsqueeze(0).to(self._device)
                )
                next_state, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                self._memory_replay.rpc_sync().push(
                    torch.Tensor(np.array(state)),
                    action,
                    reward,
                    torch.Tensor(np.array(next_state)),
                    1 * done)
                state = next_state

                if n_steps is not None and step >= n_steps:
                    return


# TODO: merge Coordinator and Parameter Server
class Coordinator:
    def __init__(self,
                 parameter_server: ParameterServer,
                 memory_buffer: rpc.RRef,
                 actors: List[rpc.RRef],
                 learners: List[rpc.RRef]):
        self._parameter_server = parameter_server
        self._memory_buffer = memory_buffer
        self._actors = actors
        self._learners = learners

    def train(self, n_frames_before_training: int, steps_per_learner: int):
        for learner in self._learners:
            self._parameter_server.subscribe(learner)
        n_frames_before_training = ceil(n_frames_before_training / len(self._actors))

        async_calls = []
        for actor in self._actors:
            async_calls.append(actor.rpc_async(timeout=0).act(n_frames_before_training))
        for call in async_calls:
            call.wait()

        async_calls = []
        for learner in self._learners:
            async_calls.append(learner.rpc_async(timeout=0).train(steps_per_learner))
        for actor in self._actors:
            actor.rpc_async(timeout=0).act()
        for call in async_calls:
            call.wait()

        async_calls = []
        for actor in self._actors:
            async_calls.append(actor.rpc_async().stop())
        for call in async_calls:
            call.wait()

        return self._parameter_server.result()
