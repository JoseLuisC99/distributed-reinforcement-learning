import asyncio
import os.path
from threading import Lock
import logging

import grpc
import torch
from torch.distributions import Categorical

from distrl.a3c.a3c_pb2 import *
from distrl.grpc.tools import *
from torch import nn
from torch import optim
import gymnasium as gym
from typing import Union, Tuple

logger = logging.getLogger(__name__)


class Coordinator(CoordinatorServicer):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 lr: float,
                 update_every: int,
                 n_steps: int,
                 steps_per_bootstrap: int,
                 max_version_tol: int = 0,
                 checkpoint_freq: Optional[int] = None,
                 output: Optional[str] = None):
        self.actor = actor
        self.critic = critic
        self.lock = WritePreferringLock()
        self.actor_gradients = [None] * update_every
        self.critic_gradients = [None] * update_every
        self.next_grad_slot = 0
        self.version = 0
        self.n_steps = n_steps
        self.n_steps_elapsed = 0
        self.steps_per_bootstrap = steps_per_bootstrap
        self.max_version_tol = max_version_tol
        self.checkpoint_freq = checkpoint_freq
        self.server = None
        self.output = output

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def PushGradients(self, request, context):
        self.lock.acquire_write_lock()
        if abs(request.version - self.version) > self.max_version_tol:
            context.set_code(grpc.StatusCode.OUT_OF_RANGE)
            context.set_details(f"mismatch version: current version is {self.version}")
        else:
            idx = self.next_grad_slot
            self.next_grad_slot += 1
            self.actor_gradients[idx] = parse_tensor_parameters(request.actor)
            self.critic_gradients[idx] = parse_tensor_parameters(request.critic)

            self.n_steps_elapsed += self.steps_per_bootstrap

            if self.next_grad_slot == len(self.actor_gradients):
                self.update()
                self.next_grad_slot = 0
            if self.n_steps_elapsed >= self.n_steps:
                logger.info(f"training finished after {self.n_steps_elapsed} global steps, shutdown coordinator")
                self.server.stop(5)

                if self.output is not None:
                    torch.save(self.actor.state_dict(), os.path.join(self.output, f"actor.pt"))
                    torch.save(self.critic.state_dict(), os.path.join(self.output, f"critic .pt"))

                    logger.info(f"saving actor model in {os.path.join(self.output, f'actor.pt')}")
                    logger.info(f"saving critic model in {os.path.join(self.output, f'critic.pt')}")
        self.lock.release_write_lock()
        return VersionMessage(version=self.version)

    def Synchronize(self, request, context):
        self.lock.acquire_read_lock()

        if self.n_steps_elapsed >= self.n_steps:
            response = ParametersMessage(terminated=True)
        elif request.version == self.version:
            response = ParametersMessage(version=request.version)
        else:
            actor_params = dict()
            critic_params = dict()
            for name, param in self.actor.named_parameters():
                actor_params[name] = tensor2proto(param.detach().cpu().data)
            for name, param in self.critic.named_parameters():
                critic_params[name] = tensor2proto(param.detach().cpu().data)
            response = ParametersMessage(actor=actor_params, critic=critic_params, version=self.version)
        self.lock.release_read_lock()
        return response

    def start(self, max_workers: int, port: int):
        try:
            logger.info(f"coordinator listening on port {port}")
            self.server = create_server(add_CoordinatorServicer_to_server, max_workers, port, self)
            self.server.start()
            self.server.wait_for_termination()
        except RuntimeError as e:
            logger.error(str(e))

    def update(self):
        for param in self.actor.parameters():
            param.grad = torch.zeros_like(param)
        for param in self.critic.parameters():
            param.grad = torch.zeros_like(param)

        # print(self.actor_gradients[0])
        for name, param in self.actor.named_parameters():
            param.grad.data += sum(map(lambda x: x[name], self.actor_gradients)) / len(self.actor_gradients)
        for name, param in self.critic.named_parameters():
            param.grad.data += sum(map(lambda x: x[name], self.critic_gradients)) / len(self.actor_gradients)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.version += 1

        if self.checkpoint_freq is not None and self.version % self.checkpoint_freq == 0:
            torch.save(self.actor.state_dict(), os.path.join("checkpoints", f"actor_{self.version}.pt"))
            torch.save(self.critic.state_dict(), os.path.join("checkpoints", f"critic_{self.version}.pt"))

        logger.debug(f"new version {self.version}")


class Worker:
    def __init__(self,
                 coordinator_addr: str,
                 coordinator_port: int,
                 actor: nn.Module,
                 critic: nn.Module,
                 env: Union[gym.Wrapper, gym.Env],
                 device: torch.device = torch.device("cpu")):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.version = 0
        self.lock = Lock()
        self.channel = grpc.insecure_channel(f"{coordinator_addr}:{coordinator_port}")
        self.coordinator = CoordinatorStub(self.channel)
        self.env = env
        self.device = device

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def run(self, gamma: float, n_steps: int):
        t = 0
        state, _ = self.env.reset()
        state = torch.Tensor(np.array(state))

        while True:
            # Reset gradients
            for param in self.actor.parameters():
                param.grad = torch.zeros_like(param)
            for param in self.critic.parameters():
                param.grad = torch.zeros_like(param)

            # Synchronize the parameters
            sync_response = None
            try:
                sync_response = self.coordinator.Synchronize(VersionMessage(version=self.version))
            except grpc.RpcError as e:
                logger.warning(f"unable to synchronize actor and critic networks. {e.code().name} - {e.details()}")
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.info(f"shutdown worker")
                    return

            if sync_response.terminated:
                logger.info(f"terminating worker after {t} iterations")
                return
            elif sync_response.version != self.version:
                self.version = sync_response.version
                for name, param in self.actor.named_parameters():
                    param.data = proto2tensor(sync_response.actor[name])
                for name, param in self.critic.named_parameters():
                    param.data = proto2tensor(sync_response.critic[name])
                logger.debug(f"parameters updated, current version is {self.version}")

            done = False
            t_start = t
            rewards = []
            states = []
            log_probs = []
            while not done and t - t_start != n_steps:
                state = state.unsqueeze(0).to(self.device)
                action, log_prob = self.get_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                state = torch.Tensor(np.array(state))

                rewards.append(reward)
                states.append(state)
                log_probs.append(log_prob)

                done = truncated or terminated
                t += 1

            r = 0 if done else self.critic(state)
            actor_loss = torch.tensor([0.0])
            critic_loss = torch.tensor([0.0])
            for ri, si, logi in zip(reversed(rewards), reversed(states), reversed(log_probs)):
                r = ri + gamma * r
                v = self.critic(si)
                tmp_diff = r - v
                actor_loss += logi * tmp_diff.clone().detach()
                critic_loss += tmp_diff ** 2
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)

            actor_grads = dict()
            critic_grads = dict()
            for name, param in self.actor.named_parameters():
                actor_grads[name] = tensor2proto(param.grad.detach().cpu().data)
            for name, param in self.critic.named_parameters():
                critic_grads[name] = tensor2proto(param.grad.detach().cpu().data)
            try:
                self.coordinator.PushGradients(ParametersMessage(
                    actor=actor_grads, critic=critic_grads, version=self.version
                ))
            except grpc.RpcError as e:
                logger.warning(f"gradients rejected. {e.code().name} - {e.details()}")

                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.info(f"shutdown worker")
                    return

            if done:
                state, _ = self.env.reset()
                state = torch.Tensor(np.array(state))
