import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

from models import Policy
from utils import to_tensor

class VPG:
    def __init__(self, args):
        self.device = torch.device("cuda") if args.cuda else torch.device("cpu")
        self.policy = Policy().to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.num_traj = args.num_traj
        self.samples = []

    def create_samples(self, env):
        trajectory_lengths = []
        for sample_idx in range(self.num_traj):
            one_traj = []
            state = env.reset()
            done = False
            while not done:
                softmax_value = self.policy(to_tensor(state).to(self.device)).detach()
                action = Categorical(softmax_value).sample().item()
                next_state, reward, done, _ = env.step(action)
                one_traj.append([state, action, reward, 0.]) # last 0 for reward to go
                state = next_state
            # Calculate and Add reward to go
            reward_sum = 0.
            for i in range(len(one_traj)-1, -1, -1):
                reward_sum += one_traj[i][2]
                one_traj[i][3] = reward_sum
            # save trajectory as a sample
            self.samples.extend(one_traj)
            trajectory_lengths.append(len(one_traj))
        mean_traj_len = sum(trajectory_lengths) / len(trajectory_lengths)
        return mean_traj_len
    
    def update_parameters(self):
        state_batch, action_batch, _, reward_to_go_batch = map(np.stack, zip(*self.samples))
        
        state_batch = to_tensor(state_batch).to(self.device)
        action_batch = to_tensor(action_batch).to(torch.long).to(self.device)
        reward_to_go_batch = to_tensor(reward_to_go_batch).to(self.device)
        pi_st_at = self.policy(state_batch)[range(len(state_batch)), action_batch]
        log_pi = torch.log(pi_st_at)
        loss = -(torch.sum(log_pi * reward_to_go_batch) / self.num_traj)
        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        # Off policy
        self.samples = []

        return loss