import logging
import os
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Literal

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformReward, FrameStack
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
    def __init__(self, n_actions: int):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, n_actions)

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.type(torch.float32) / 255.0
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(nn.Flatten()(x)), 0.01)
        return self.fc2(x)


def get_arguments() -> Namespace:
    args_parser = ArgumentParser(description="GORILA")
    args_parser.add_argument("--master-addr",
                             type=str, default="localhost")
    args_parser.add_argument("--master-port",
                             type=int, default=8080)
    args_parser.add_argument("--env",
                             type=str, default="ALE/Breakout-v5")
    args_parser.add_argument("--batch-size",
                             type=int, default=32)
    args_parser.add_argument("--gamma",
                             type=float, default=0.99)
    args_parser.add_argument("--steps",
                             type=int, default=1_000_000)
    args_parser.add_argument("--mem-size",
                             type=int, default=100_000)
    args_parser.add_argument("--learning-rate",
                             type=float, default=0.00025)
    args_parser.add_argument("--sync-freq",
                             type=int, default=60_000)
    args_parser.add_argument("--max-version-diff",
                             type=int, default=20)
    args_parser.add_argument("--checkpoint",
                             type=bool, required=False, default=True)
    args_parser.add_argument("--output",
                             type=str, required=False, default=None)
    args_parser.add_argument("--log",
                             type=str, required=False, default="INFO")
    args_parser.add_argument("--annealing",
                             type=int, default=1_000_000)
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

    env = gym.make(args.env, frameskip=1)
    env = AtariPreprocessing(env, scale_obs=False)
    env = FrameStack(env, 4, True)
    env = TransformReward(env, lambda x: min(max(x, -1.0), 1.0))

    action_space_dim = env.action_space.n
    observation_shape = env.observation_space.shape

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

        annealer = LinearAnnealer(1.0, 0.1, args.annealing)
        ps = ParameterServer(
            DeepQNetwork(action_space_dim),
            actors,
            learners,
            args.learning_rate,
            args.sync_freq,
            annealer,
            args.max_version_diff,
            args.checkpoint
        )

        target_network = ps.train(args.mem_size)
        if args.output is not None:
            torch.save(target_network.state_dict(), os.path.join(args.output, f"atari-gorila.pth"))
            logger.info(f'model saved on {os.path.abspath(os.path.join(args.output, f"atari-gorila.pth"))}')
    elif node_type == "actor":
        rref_ps = rpc.remote(0, get_ps)
        rref_memory = rpc.remote(1, get_memory_buffer)
        actor = Actor(
            rank,
            env,
            rref_memory,
            rref_ps,
            DeepQNetwork(action_space_dim),
            device
        )
    elif node_type == "learner":
        rref_ps = rpc.remote(0, get_ps)
        rref_memory = rpc.remote(1, get_memory_buffer)
        learner = Learner(
            rank,
            DeepQNetwork(action_space_dim),
            rref_ps,
            rref_memory,
            args.batch_size,
            args.gamma,
            device
        )
    elif node_type == "memory":
        memory_buffer = MemoryReplay(
            args.mem_size, observation_shape, args.steps,
            obs_dtype=torch.int8, action_dtype=torch.int8, rewards_dtype=torch.int8
        )
        memory_buffer.subscribe(rpc.remote(0, get_ps))
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

    print("#" * 50)

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
