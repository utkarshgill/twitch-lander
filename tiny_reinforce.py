"""
REINFORCE from scratch (~130 lines)

1. policy π(a|s) = tanh-squashed gaussian
2. rollout episodes, compute returns G_t = Σ γ^k r_{t+k}
3. update: maximize E[log π(a|s) * G]
4. repeat
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from collections import deque
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import os

PLOT = int(os.getenv('PLOT', '0'))

class Policy(nn.Module):
    def __init__(self, in_dim=8, out_dim=2):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), 
                                nn.Linear(128, 128), nn.ReLU(),
                                nn.Linear(128, out_dim))
        self.log_std = nn.Parameter(torch.zeros(out_dim) - 1.0)  # init std = exp(-1) ≈ 0.37
    
    def __call__(self, x):
        mu = self.mu(x)
        std = self.log_std.exp()
        u = Normal(mu, std).sample()
        a = torch.tanh(u)  # squash to [-1, 1]
        return a, u, mu, std

def log_prob(u, mu, std):
    # we sampled u from gaussian, but sent a=tanh(u) to env
    # change of variables: log P(a) = log P(u) - log|da/du| = log P(u) - log(1-a²)
    a = torch.tanh(u)
    logp_u = Normal(mu, std).log_prob(u).sum(-1)
    logp_a = logp_u - torch.log(1 - a**2 + 1e-6).sum(-1)
    return logp_a

def returns(rewards, gamma=0.99):
    # G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    G = []
    running = 0
    for r in reversed(rewards):
        running = r + gamma * running
        G.append(running)
    return G[::-1]

@torch.no_grad()
def rollout(env, pi):
    s, _ = env.reset()
    trajectory = []
    done = False
    obs_scale = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1], dtype=np.float32)  # scale obs for nn
    
    while not done:
        s_t = torch.FloatTensor(s * obs_scale)
        a, u, _, _ = pi(s_t)
        s, r, term, trunc, _ = env.step(a.numpy())
        trajectory.append((s_t, u, r))  # save state, action_sample, reward
        done = term or trunc
    
    return trajectory

def update(pi, opt, trajectories, gamma=0.99):
    # flatten all episodes into one batch
    batch = []
    for traj in trajectories:
        G = returns([step[2] for step in traj], gamma)
        for (s, u, r), g in zip(traj, G):
            batch.append((s, u, g))
    
    states = torch.stack([x[0] for x in batch])
    us = torch.stack([x[1] for x in batch])
    Gs = torch.FloatTensor([x[2] for x in batch])
    
    # recompute log probs with gradients (rollout was @no_grad)
    mus = pi.mu(states)
    stds = pi.log_std.exp().expand_as(mus)
    logps = log_prob(us, mus, stds)
    
    # REINFORCE: maximize E[log π * (G - baseline)]
    advantages = Gs - Gs.mean()  # baseline for variance reduction
    loss = -(logps * advantages).mean()
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
    opt.step()
    
    return loss.item(), Gs.mean().item()

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous=True)
    pi = Policy()
    opt = torch.optim.Adam(pi.parameters(), lr=3e-4)
    
    best_reward = -float('inf')
    reward_history = deque(maxlen=100)
    all_rewards = []
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))
    
    tqdm.write("iter | r_batch | r_100 | r_best")
    tqdm.write("-" * 40)
    
    for i in trange(1000, desc="training", ncols=80, leave=True):
        trajs = [rollout(env, pi) for _ in range(10)]
        ep_rewards = [sum(step[2] for step in traj) for traj in trajs]
        
        loss, _ = update(pi, opt, trajs)
        
        reward = np.mean(ep_rewards)
        reward_std = np.std(ep_rewards)
        reward_history.append(reward)
        smooth = np.mean(reward_history)
        all_rewards.append(reward)
        
        is_best = reward > best_reward
        if is_best:
            best_reward = reward
        
        # print on new best or eval
        if is_best:
            tqdm.write(f"{i:4d} | {reward:6.1f} | {smooth:6.1f} | {best_reward:5.1f} 🚀 (±{reward_std:.1f})")
        else:
            tqdm.write(f"{i:4d} | {reward:6.1f} | {smooth:6.1f} | {best_reward:5.1f}")
        
        if PLOT and i % 5 == 0:
            ax.clear()
            ax.plot(all_rewards, alpha=0.3, label='episode avg', color='blue')
            ax.plot([np.mean(all_rewards[max(0,j-100):j+1]) for j in range(len(all_rewards))], 
                   label='rolling 100', linewidth=2, color='orange')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='solved threshold')
            ax.set_xlabel('iteration')
            ax.set_ylabel('reward')
            ax.set_title('REINFORCE on LunarLander-v3')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.pause(0.001)
        
        if i % 100 == 0:
            env_vis = gym.make("LunarLander-v3", continuous=True, render_mode="human")
            eval_reward = sum(x[2] for x in rollout(env_vis, pi))
            env_vis.close()
            tqdm.write(f"{'='*40}")
            tqdm.write(f"EVAL: {eval_reward:.1f} | batch: {reward:.1f}±{reward_std:.1f}")
            tqdm.write(f"{'='*40}")
    
    env.close()
    if PLOT:
        plt.ioff()
        plt.show()
    tqdm.write(f"\n✅ done. best: {best_reward:.1f}")
