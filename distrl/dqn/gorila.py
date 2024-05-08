import copy
import logging
import math
import os
import random
import threading
import time
from itertools import count
from math import ceil
from typing import Any, Optional, Dict, Tuple, List, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributed import rpc

from distrl.dqn.utils import LinearAnnealer

logger = logging.getLogger(__name__)


class ParameterServer:
    def __init__(self,
                 q_network: nn.Module,
                 actors: List[rpc.RRef],
                 learners: List[rpc.RRef],
                 lr: float,
                 sync_frequency: int,
                 annealer: LinearAnnealer,
                 max_version_difference: Optional[int] = None,
                 checkpoint: bool = True):
        self._q_network = q_network
        self._target_network = copy.deepcopy(q_network)
        self._actors = actors
        self._learners = learners
        self._annealer = annealer
        self.__version = 0
        self._optimizer = optim.Adam(self._q_network.parameters(), lr=lr, eps=1.5e-4)
        self._max_version_difference = max_version_difference
        self._sync_frequency = sync_frequency
        self._checkpoint = checkpoint
        self.__listeners = []
        self.__lock = threading.Lock()
        self.__start_time = 0
        self.__stop_signal = False

        if self._checkpoint:
            os.makedirs("checkpoints", exist_ok=True)
            logger.info("`checkpoint` dir created")

    def train(self, n_frames_before_training: int):
        self.__start_time = time.time()
        for learner in self._learners:
            self.subscribe(learner)
        for actor in self._actors:
            self.subscribe(actor)

        logger.info("filling memory buffer")
        n_frames_before_training = ceil(n_frames_before_training / len(self._actors))
        async_calls = []
        for actor in self._actors:
            async_calls.append(actor.rpc_async(timeout=0).act(n_steps=n_frames_before_training))
        for call in async_calls:
            call.wait()

        logger.info(f"staring {len(self._learners)} learners")
        self.sync_target_network()
        async_calls = []
        for learner in self._learners:
            async_calls.append(learner.rpc_async(timeout=0).train())
        logger.debug(f"starting {len(self._actors)} actors.")
        for actor in self._actors:
            async_calls.append(actor.rpc_async(timeout=0).act(annealer=copy.copy(self._annealer)))
        for call in async_calls:
            call.wait()

        return self._target_network

    def update(self, learner: int, version: int, gradients: Dict[str, torch.Tensor]) -> bool:
        with self.__lock:
            if abs(self.__version - version) > self._max_version_difference:
                logger.warning(f"rejecting gradient from learner {learner}: current version is {self.__version}, but "
                               f"receives version {version}")
                return False

            self.__version += 1
            self._optimizer.zero_grad()
            for name, param in self._q_network.named_parameters():
                if param.requires_grad:
                    param.grad = gradients[name]
            self._optimizer.step()

            if self.__version % self._sync_frequency == 0:
                logger.info(f"Synchronizing new target network. Current version {self.__version}")
                self.sync_target_network()

        return True

    def parameters(self) -> Tuple[int, Dict[str, Any]]:
        with self.__lock:
            return self.__version, self._q_network.state_dict()

    def subscribe(self, rref: rpc.RRef):
        logger.info(f"New node added: {rref.owner()}")
        self.__listeners.append(rref)

    def sync_target_network(self) -> None:
        version = self.__version
        state_dict = self._q_network.state_dict()
        self._target_network.load_state_dict(state_dict)
        futures = []
        for rref in self.__listeners:
            futures.append(rref.rpc_async().sync(state_dict, version))
        for fut in futures:
            fut.wait()

        if self._checkpoint:
            torch.save(state_dict, os.path.join("checkpoints", f"checkpoint-{version}.pth"))

    def stop(self):
        end = time.time()
        futures = []
        for rref in self.__listeners:
            futures.append(rref.rpc_async().stop())
        for fut in futures:
            fut.wait()
        logger.info(f"end of training after {math.ceil(end - self.__start_time)} seconds")


class Learner:
    def __init__(self,
                 rank_id: int,
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
        self.__rank_id = rank_id

    def sync(self, state_dict: Dict[str, Any], version: int):
        with self.__lock:
            self._target_network.load_state_dict(state_dict)

    def stop(self):
        logger.warning(f"stopping learner {self.__rank_id} with version {self.__version}")
        self.__stop_signal = True

    def train(self):
        while not self.__stop_signal:
            with self.__lock:
                version, state_dict = self._parameter_server.rpc_sync().parameters()
                self._q_network.load_state_dict(state_dict)
                self.__version = version
            logger.debug(f"loading Q-network on learner {self.__rank_id}: version {self.__version}")

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
            if not self._parameter_server.rpc_sync().update(self.__rank_id, self.__version, gradients):
                pass


class Actor:
    def __init__(self,
                 rank_id: int,
                 env: Union[gym.Wrapper, gym.Env],
                 memory_replay: rpc.RRef,
                 parameter_server: rpc.RRef,
                 q_network: nn.Module,
                 device: torch.device = torch.device("cpu")):
        self._env = env
        self._memory_replay = memory_replay
        self._parameter_server = parameter_server
        self._device = device
        self._q_network = q_network.to(self._device)
        self.__stop_signal = False
        self._annealer = None

        self.__n_samples = 0
        self.__rank_id = rank_id
        self.__version = 0

        self.__lock = threading.Lock()

    def sync(self, state_dict: Dict[str, Any], version: int):
        if self._annealer is not None:
            logger.info(f"Synchronizing actor {self.__rank_id}: current epsilon value "
                        f"is {self._annealer.current_value} after {self._annealer.current_step} steps")
        with self.__lock:
            self._q_network.load_state_dict(state_dict)
            self.__version = version

    def _epsilon_greedy(self, epsilon: float, state: torch.Tensor) -> Any:
        assert 0.0 <= epsilon <= 1.0
        if random.random() < epsilon:
            action = self._env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self._q_network(state)
                action = torch.argmax(q_values, dim=1).detach().cpu().item()
        return action

    def stop(self):
        logger.warning(f"stopping actor {self.__rank_id} with version {self.__version}. {self.__n_samples} samples "
                       f"generated.")
        self.__stop_signal = True

    def act(self, annealer: Optional[LinearAnnealer] = None, n_steps: Optional[int] = None):
        step = 0
        self._annealer = annealer
        for _ in count():
            state, _ = self._env.reset()
            done = False

            while not done:
                self.__n_samples += 1
                if self.__stop_signal:
                    return

                step += 1
                epsilon = 1.0 if self._annealer is None else self._annealer.step()
                action = self._epsilon_greedy(
                    epsilon, torch.Tensor(np.array(state)).unsqueeze(0).to(self._device)
                )
                next_state, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated

                try:
                    self._memory_replay.rpc_sync().push(
                        torch.Tensor(np.array(state)),
                        action,
                        reward,
                        torch.Tensor(np.array(next_state)),
                        done)
                except RuntimeError:
                    logger.error(f"memory buffer disconnected, aborting actor {self.__rank_id} after "
                                 f"{self.__n_samples} samples")
                    return
                state = next_state

                if n_steps is not None and step >= n_steps:
                    return
