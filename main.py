import argparse
import random

import gym
import torch
import numpy as np
import wandb

from vpg import VPG


def main():
    parser = argparse.ArgumentParser(description="Vanilla Policy Gradient")
    parser.add_argument("--env_name",
                        default="CartPole-v1",
                        help="Environment name (default: CartPole-v1)")
    parser.add_argument("--num_traj",
                        default=1000,
                        type=int,
                        metavar='N',
                        help="Number of trajectories to use as sample (default: 1000)")
    parser.add_argument("--lr",
                        default=0.003,
                        type=float,
                        metavar='G',
                        help="Learning rate (default: 0.003)")
    parser.add_argument("--epoch",
                        default=200,
                        type=int,
                        metavar='N',
                        help="Number of Epochs (default: 200)")
    parser.add_argument("--seed",
                        default=123456,
                        type=int,
                        metavar='N',
                        help="Random seed (default: 123456)")
    parser.add_argument("--cuda", action="store_true", help="Whether to use CUDA (default: False)")
    parser.add_argument('--wandb',
                        action="store_true",
                        help='Whether use Weight and Bias for logging(default: False)')
    parser.add_argument('--wandb_id', default=None, help='ID for wandb account(default: None)')
    parser.add_argument('--wandb_project',
                        default=None,
                        help='project name of wandb account(default: None)')

    args = parser.parse_args()
    env = gym.make(args.env_name)
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config = {
            'env_name': args.env_name,
            'num_traj': args.num_traj,
            'epoch': args.epoch,
            'seed': args.seed,
            'cuda': args.cuda,
            'wandb': args.wandb,
            'wandb_id': args.wandb_id,
            'wandb_project': args.wandb_project,
        }
    # setting seed
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = VPG(args)

    for ith_epoch in range(1, args.epoch + 1):
        mean_traj_len = agent.create_samples(env)
        loss = agent.update_parameters()
        print(f'Epoch: {ith_epoch}, Mean Trajectory Length: {mean_traj_len}')
        if args.wandb:
            wandb.log({
                'mean_traj_len': mean_traj_len,
                'loss': loss,
            })
        if ith_epoch % 20 == 0:
            agent.save_model(epoch=ith_epoch)


if __name__ == "__main__":
    main()