import logging
import os
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Literal

import gymnasium as gym
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed import rpc

from distrl.dqn.gorila import ParameterServer, Learner, Actor
from distrl.dqn.utils import MemoryReplay, LinearAnnealer

logger = logging.getLogger(__name__)

ps = None
memory_buffer = None
actor = None
learner = None


def get_memory_buffer(retry: int = 5, max_attempts: int = 4):
    global memory_buffer
    while memory_buffer is None:
        logger.warning(f"memory buffer not initialized, waiting for {retry}s")
        time.sleep(retry)
        max_attempts -= 1
        if max_attempts <= 0:
            raise Exception("memory buffer not initialized")
    return memory_buffer


def get_actor(retry: int = 5, max_attempts: int = 4):
    global actor
    while actor is None:
        logger.warning(f"actor not initialized, waiting for {retry}s")
        time.sleep(retry)
        max_attempts -= 1
        if max_attempts <= 0:
            raise Exception("actor not initialized")
    return actor


def get_learner(retry: int = 5, max_attempts: int = 4):
    global learner
    while learner is None:
        logger.warning(f"learner not initialized, waiting for {retry}s")
        time.sleep(retry)
        max_attempts -= 1
        if max_attempts <= 0:
            raise Exception("learner not initialized")
    return learner


def get_ps(retry: int = 5, max_attempts: int = 4):
    global ps
    while ps is None:
        logger.warning(f"parameter server not initialized, waiting for {retry}s")
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
    args_parser.add_argument("--master-addr",
                             type=str, default="localhost")
    args_parser.add_argument("--master-port",
                             type=int, default=8080)
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
    args_parser.add_argument("--output",
                             type=str, required=False, default=None)
    args_parser.add_argument("--log",
                             type=str, required=False, default="INFO")
    args_parser.add_argument("--annealing",
                             type=int, default=20_000)
    args_parser.add_argument("--n-actors", type=int)
    args_parser.add_argument("--n-learners", type=int)

    return args_parser.parse_args()


def run_node(
        rank: int,
        world_size: int,
        node_type: Literal["ps", "actor", "learner", "memory"],
        args: Namespace, *,
        cuda: bool = False,
        num_worker_threads: int = 32):
    global ps
    global memory_buffer
    global actor
    global learner

    device = torch.device("cuda" if cuda else "cpu")
    if cuda and not torch.cuda.is_available():
        logger.error(f"CUDA device not available for node_{rank}")
    elif not cuda and torch.cuda.is_available():
        device = torch.device("cpu")
    logger.info(f"using {device.type} device in node_{rank}")

    env = gym.make(args.env)

    action_space_dim = 2
    observation_shape = 4

    rpc.init_rpc(
        f"node_{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=num_worker_threads, _transports=["uv"])
    )

    if node_type == "ps":
        rref_memory = rpc.remote(1, get_memory_buffer)
        learners = [rpc.remote(remote_id, get_learner) for remote_id in range(2, 2 + args.n_learners)]
        actors = [
            rpc.remote(remote_id, get_actor)
            for remote_id in range(2 + args.n_learners, 2 + args.n_learners + args.n_actors)]

        ps = ParameterServer(
            DeepQNetwork(observation_shape, action_space_dim),
            rref_memory,
            actors,
            learners,
            args.learning_rate,
            args.sync_freq,
            args.max_version_diff,
            args.checkpoint
        )

        target_network = ps.train(args.mem_size, args.steps)
        if args.output is not None:
            torch.save(target_network.state_dict(), args.output)
    elif node_type == "actor":
        rref_ps = rpc.remote(0, get_ps)
        rref_memory = rpc.remote(1, get_memory_buffer)
        # annealer = LinearAnnealer(1.0, 0.1, args.steps)
        annealer = LinearAnnealer(1.0, 0.1, args.annealing)
        actor = Actor(
            rank,
            env,
            rref_memory,
            rref_ps,
            DeepQNetwork(observation_shape, action_space_dim),
            annealer,
            args.sync_freq,
            device
        )
    elif node_type == "learner":
        rref_ps = rpc.remote(0, get_ps)
        rref_memory = rpc.remote(1, get_memory_buffer)
        learner = Learner(
            rank,
            DeepQNetwork(observation_shape, action_space_dim),
            rref_ps,
            rref_memory,
            args.batch_size,
            args.gamma,
            device
        )
    elif node_type == "memory":
        memory_buffer = MemoryReplay(args.mem_size, (observation_shape, ))
        logger.debug("memory buffer ready")

    rpc.shutdown()


if __name__ == "__main__":
    args = get_arguments()

    if args.log is not None:
        os.makedirs("logs", exist_ok=True)
        log_level = getattr(logging, args.log.upper(), None)
        if not isinstance(log_level, int):
            raise ValueError(f"Invalid log level: {args.log}")
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
            filename=os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d %H:%M:00')}.log"),
            encoding="utf-8",
            level=log_level
        )

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    world_size = 2 + args.n_learners + args.n_actors
    processes = []

    ps_process = mp.Process(target=run_node, args=(0, world_size, "ps", args), kwargs={"cuda": False})
    memory_process = mp.Process(target=run_node, args=(1, world_size, "memory", args), kwargs={"cuda": False})

    ps_process.start()
    memory_process.start()

    processes.append(ps_process)
    processes.append(memory_process)

    n_processes = len(processes)
    for i in range(args.n_learners):
        p = mp.Process(target=run_node, args=(n_processes + i, world_size, "learner", args), kwargs={"cuda": True})
        p.start()
        processes.append(p)

    n_processes = len(processes)
    for i in range(args.n_actors):
        p = mp.Process(target=run_node, args=(n_processes + i, world_size, "actor", args), kwargs={"cuda": True})
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
