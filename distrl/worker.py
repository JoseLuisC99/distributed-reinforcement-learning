import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.distributed as dist
import gymnasium as gym
import logging
from collections import OrderedDict
import time


logger = logging.getLogger("Workers")
logger.setLevel(logging.INFO)


class ModelDistBuffer:
    def __init__(self, model: nn.Module):
        self.version = 0
        self.__parameter_buffer = {}
        self.__model = model

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.__parameter_buffer[name] = torch.empty(param.size(), dtype=param.dtype)

    def fetch(self):
        for param in self.__parameter_buffer:
            dist.broadcast(self.__parameter_buffer[param], src=0)
        self.version += 1
        return self.__parameter_buffer


class DistAgent:
    def __init__(self, rank: int,
                 policy: nn.Module,
                 env: str,
                 max_iters: int,
                 gamma: float,
                 device: torch.device = torch.device("cpu")):
        self._rank = rank
        self._device = device
        self._max_iters = max_iters
        self._gamma = gamma

        self.policy = policy.to(self._device)
        self.env = gym.make(env)
        self.running_reward = 0
        self._model_buffer = ModelDistBuffer(self.policy)

        self.__rewards = []
        self.__actions = []

    def update(self):
        new_params = self._model_buffer.fetch()
        for name in new_params:
            new_params[name].to(self._device)
        self.policy.load_state_dict(OrderedDict(new_params))

    def send_grads(self):
        for name, param in self.policy.named_parameters():
            if self._device.type == "cuda":
                grad = param.grad.to(torch.device("cpu")).detach()
            else:
                grad = param.grad.detach()
            dist.gather(grad, dst=0)

    def select_action(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.__actions.append(m.log_prob(action))
        return action.item()

    def run_episode(self):
        logger.info(f"Running episode on worker {self._rank}.")

        update_time = time.time()
        self.update()
        update_time = time.time() - update_time

        episode_time = time.time()
        state, _ = self.env.reset()
        for _ in range(self._max_iters):
            action = self.select_action(state)
            state, reward, done, _, _ = self.env.step(action)
            self.__rewards.append(reward)
            if done:
                break
        episode_time = time.time() - episode_time

        reward = sum(self.__rewards)
        self.running_reward = 0.05 * reward + (1 - 0.05) * self.running_reward

        R = 0
        returns = []
        for r in self.__rewards[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 0.001)

        loss = []
        for log_prob, R in zip(self.__actions, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()
        loss.backward()

        grads_time = time.time()
        self.send_grads()
        grads_time = time.time() - grads_time

        del self.__actions[:]
        del self.__rewards[:]

        logger.info((f"Iteration {self._model_buffer.version} on worker {self._rank}: ",
                     f"reward {reward:.4f}  | ",
                     f"running_reward {self.running_reward:.4f} | ",
                     f"update model time {update_time:.4f}s | ",
                     f"episode time {episode_time:.4f}s | ",
                     f"sending gradients time {grads_time:.4f}s"))

        return reward
