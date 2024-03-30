import os
import time
from argparse import ArgumentParser, Namespace

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import rpc

from distrl.dqn.gorila import ParameterServer, Learner, Actor, Coordinator
from distrl.dqn.utils import MemoryReplay, LinearAnnealer

ps = None
memory_buffer = None
actor = None
learner = None


def get_memory_buffer(retry: int = 5, max_attempts: int = 4):
    global memory_buffer
    while memory_buffer is None:
        print(f"memory buffer not initialized, waiting for {retry}s")
        time.sleep(retry)
        max_attempts -= 1
        if max_attempts <= 0:
            raise Exception("memory buffer not initialized")
    return memory_buffer


def get_actor(retry: int = 5, max_attempts: int = 4):
    global actor
    while actor is None:
        print(f"actor not initialized, waiting for {retry}s")
        time.sleep(retry)
        max_attempts -= 1
        if max_attempts <= 0:
            raise Exception("actor not initialized")
    return actor


def get_learner(retry: int = 5, max_attempts: int = 4):
    global learner
    while learner is None:
        print(f"learner not initialized, waiting for {retry}s")
        time.sleep(retry)
        max_attempts -= 1
        if max_attempts <= 0:
            raise Exception("learner not initialized")
    return learner


def get_ps(retry: int = 5, max_attempts: int = 4):
    global ps
    while ps is None:
        print(f"parameter server not initialized, waiting for {retry}s")
        time.sleep(retry)
        max_attempts -= 1
        if max_attempts <= 0:
            raise Exception("parameter server not initialized")
    return ps


class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def get_arguments() -> Namespace:
    args_parser = ArgumentParser(description="GORILA")
    # RPC information
    args_parser.add_argument("--rank",
                             type=int, required=True)
    args_parser.add_argument("--world-size",
                             type=int, required=True)
    args_parser.add_argument("--node-type",
                             type=str, choices=["ps", "actor", "learner", "memory"], required=True)
    args_parser.add_argument("--master-addr",
                             type=str, default="localhost")
    args_parser.add_argument("--master-port",
                             type=int, default=8080)

    # Runtime configuration
    args_parser.add_argument("--env",
                             type=str, default="CartPole-v1")
    args_parser.add_argument("--batch-size",
                             type=int, default=64)
    args_parser.add_argument("--gamma",
                             type=float, default=0.99)
    args_parser.add_argument("--steps",
                             type=int, default=20_000)
    args_parser.add_argument("--mem-size",
                             type=int, default=50_000)
    args_parser.add_argument("--learning-rate",
                             type=float, default=2e-4)
    args_parser.add_argument("--sync-freq",
                             type=int, default=1_000)
    args_parser.add_argument("--max-version-diff",
                             type=int, default=10)
    args_parser.add_argument("--checkpoint",
                             type=bool, required=False, default=True)
    args_parser.add_argument("--cuda",
                             default=False, action="store_true")
    args_parser.add_argument("--no-cuda",
                             dest="cuda", action="store_false")
    args_parser.add_argument("--output",
                             type=str, required=False, default=None)

    # Information about the other nodes
    args_parser.add_argument("--memory-rank",
                             type=int, required=False)
    args_parser.add_argument("--ps-rank",
                             type=int, required=False)
    args_parser.add_argument("--actors",
                             action='store', dest='actors', type=int, nargs='+')
    args_parser.add_argument("--learners",
                             action='store', dest='learners', type=int, nargs='+')

    return args_parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda and not torch.cuda.is_available():
        print("Warning: CUDA device not available")
    elif not args.cuda and torch.cuda.is_available():
        device = torch.device("cpu")

    env = gym.make(args.env)

    action_space_dim = 2
    observation_shape = 4

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    rpc.init_rpc(f"node_{args.rank}", rank=args.rank, world_size=args.world_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.node_type == "ps":
        ps = ParameterServer(
            DeepQNetwork(observation_shape, action_space_dim),
            args.learning_rate,
            args.sync_freq,
            args.max_version_diff,
            args.checkpoint
        )
        rref_memory = rpc.remote(args.memory_rank, get_memory_buffer)
        actors = [rpc.remote(remote_id, get_actor) for remote_id in args.actors]
        learners = [rpc.remote(remote_id, get_learner) for remote_id in args.learners]
        coordinator = Coordinator(ps, rref_memory, actors, learners)

        target_network = coordinator.train(args.mem_size, args.steps)
        if args.output is not None:
            torch.save(target_network.state_dict(), args.output)
    elif args.node_type == "actor":
        rref_ps = rpc.remote(args.ps_rank, get_ps)
        rref_memory = rpc.remote(args.memory_rank, get_memory_buffer)
        annealer = LinearAnnealer(1.0, 0.1, args.steps)
        actor = Actor(
            env,
            rref_memory,
            rref_ps,
            DeepQNetwork(observation_shape, action_space_dim),
            annealer,
            args.sync_freq,
            device
        )
    elif args.node_type == "learner":
        rref_ps = rpc.remote(args.ps_rank, get_ps)
        rref_memory = rpc.remote(args.memory_rank, get_memory_buffer)
        learner = Learner(
            DeepQNetwork(observation_shape, action_space_dim),
            rref_ps,
            rref_memory,
            args.batch_size,
            args.gamma,
            device
        )
    elif args.node_type == "memory":
        memory_buffer = MemoryReplay(args.mem_size, (observation_shape, ))
        print("memory buffer ready")

    rpc.shutdown()
