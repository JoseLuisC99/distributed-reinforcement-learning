import os
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from distrl.server import ParameterServer
from distrl.worker import DistAgent
from tqdm import tqdm
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed Reinforcement Learning Example")
    parser.add_argument("--world_size", default=4, type=int,
                        help="Number of workers.")
    parser.add_argument("--rank", type=int,
                        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="How much to value future rewards.")
    parser.add_argument("--master_addr", type=str, default="localhost",
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

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(backend='gloo', world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        policy = Policy()
        server = ParameterServer(args.world_size, policy, max_episodes=args.max_episodes)
        history = server.run(smoothing_factor=0.95)
        print(f"Global running reward: {server.running_reward:.3f}")
        if args.output_dir is not None:
            torch.save(policy.state_dict(), os.path.join(args.output_dir, f"{args.env_name}_policy.pt"))
            print("Policy model saved on", os.path.join(args.output_dir, f"{args.env_name}_policy.pt"))
        plt.plot(history)
        plt.show()
    else:
        policy = Policy()
        worker = DistAgent(policy, args.env_name, args.max_iters, args.gamma)
        for _ in tqdm(range(args.max_episodes)):
            worker.run_episode()
