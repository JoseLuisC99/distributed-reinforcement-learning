import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.distributed as dist
import gymnasium as gym
from collections import OrderedDict

class Agent:
    def __init__(self, policy: nn.Module, device: torch.device = torch.device("cpu")):
        self.policy = policy
        self._device = device

    def act(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class DistAgent(Agent):
    def __init__(self, policy: nn.Module, env: str, max_iters: int, gamma: float,
                 device: torch.device = torch.device("cpu")):
        super().__init__(policy, device)
        self._device = device
        self._max_iters = max_iters
        self._gamma = gamma

        self.policy = policy.to(self._device)
        self.env = gym.make(env)
        self.running_reward = 0

        self.__parameter_buffer = {}
        self.__rewards = []
        self.__actions = []

        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                self.__parameter_buffer[name] = torch.empty(param.size(), dtype=param.dtype)

    def fetch(self):
        for param in self.__parameter_buffer:
            dist.broadcast(self.__parameter_buffer[param], src=0)

    def update(self):
        self.fetch()
        for name in self.__parameter_buffer:
            self.__parameter_buffer[name].to(self._device)
        self.policy.load_state_dict(OrderedDict(self.__parameter_buffer))

    def send_grads(self):
        for name, param in self.policy.named_parameters():
            if self._device.type == "cuda":
                grad = param.grad.to(torch.device("cpu")).detach()
            else:
                grad = param.grad.detach()
            dist.gather(grad, dst=0)

    def send_reward(self, reward):
        dist.reduce(reward, dst=0, op=dist.ReduceOp.MIN)

    def select_action(self, state: np.ndarray):
        action, log_prob = self.act(state)
        self.__actions.append(log_prob)
        return action

    def run_episode(self):
        self.update()
        state, _ = self.env.reset()
        for _ in range(self._max_iters):
            action = self.select_action(state)
            state, reward, done, _, _ = self.env.step(action)
            self.__rewards.append(reward)
            if done:
                break

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
        self.send_grads()

        del self.__actions[:]
        del self.__rewards[:]

        self.send_reward(torch.tensor(reward))
