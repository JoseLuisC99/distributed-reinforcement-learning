import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from distrl.policy_server.server import ParameterServer
from distrl.policy_server.worker import DistAgent
from typing import Optional


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def run(rank: int, world_size: int, max_episodes: int, max_iters: int, gamma: float, master_addr: str,
        master_port: str, env_name: str, output_dir: Optional[str] = None):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)
    if rank == 0:
        print("Server started")
        policy = Policy()
        server = ParameterServer(world_size, policy, max_episodes=max_episodes)
        history = server.run(smoothing_factor=0.95)
        print(f"Global running reward: {server.running_reward:.3f}")
        if output_dir is not None:
            torch.save(policy.state_dict(), os.path.join(output_dir, f"{env_name}_{world_size - 1}workers_policy.pt"))
            print("Policy model saved on", os.path.join(output_dir, f"{env_name}_{world_size - 1}workers_policy.pt"))
        plt.plot(history)
        plt.show()
    else:
        print(f"Worker {rank} started")
        policy = Policy()
        worker = DistAgent(policy, env_name, max_iters, gamma)
        for _ in range(max_episodes):
            worker.run_episode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed Reinforcement Learning Example")
    parser.add_argument("--workers", default=4, type=int,
                        help="Number of workers.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="How much to value future rewards.")
    parser.add_argument("--master_addr", type=str, default="0.0.0.0",
                        help="Address of master, will default to localhost if not provided. Master must be able to "
                             "accept network traffic on the address + port.")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="Port that master is listening on, will default to 29500 if not provided. Master must be "
                             "able to accept network traffic on the host and port.")
    parser.add_argument("--env_name", type=str, default="CartPole-v1",
                        help="Name of the environment (see https://gymnasium.farama.org/).")
    parser.add_argument("--max_iters", type=int, default=10000,
                        help="Maximum number of iterations per period.")
    parser.add_argument("--max_episodes", type=int,
                        help="Maximum number of episodes to run.")
    parser.add_argument("--output_dir", type=str, required=False, default=None,
                        help="Directory to save the results.")
    args = parser.parse_args()

    mp.set_start_method("spawn")
    processes = []
    for rank in range(args.workers + 1):
        p = mp.Process(target=run, args=(rank, args.workers + 1, args.max_episodes, args.max_iters, args.gamma,
                                         args.master_addr, args.master_port, args.env_name, args.output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
