"""
REINFORCE from scratch (~140 lines)

Policy gradient with returns and baseline:
∇J(θ) = E_τ[Σ_t ∇log π_θ(a_t|s_t) * (G_t - b)]

where:
- π_θ(a|s) = tanh-squashed Gaussian policy
- G_t = Σ_{t'=t}^T γ^(t'-t) * r_{t'} (return from timestep t)
- b = mean baseline for variance reduction

Algorithm (epoch structure):
1. Collect experience by acting in environment until batch size reached
2. Compute returns for each timestep
3. Update policy: maximize E[log π(a|s) * (G - baseline)]
4. Repeat
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
        self.log_std = nn.Parameter(torch.zeros(out_dim))
    
    def __call__(self, x):
        mu = self.mu(x)
        std = self.log_std.exp()
        u = Normal(mu, std).sample()
        a = torch.tanh(u)  # squash to [-1, 1]
        return a, u
    
    def log_prob(self, a, u, mu, std):
        # we sampled u from gaussian, but sent a=tanh(u) to env
        # change of variables: log P(a) = log P(u) - log|da/du| = log P(u) - log(1-a²)
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

# https://gymnasium.farama.org/environments/box2d/lunar_lander/#:~:text=For%20the%20default%20values%20of%20VIEWPORT_W%2C%20VIEWPORT_H%2C%20SCALE%2C%20and%20FPS%2C%20the%20scale%20factors%20equal%3A%20%E2%80%98x%E2%80%99%3A%2010%2C%20%E2%80%98y%E2%80%99%3A%206.666%2C%20%E2%80%98vx%E2%80%99%3A%205%2C%20%E2%80%98vy%E2%80%99%3A%207.5%2C%20%E2%80%98angle%E2%80%99%3A%201%2C%20%E2%80%98angular%20velocity%E2%80%99%3A%202.5
OBS_SCALE = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1], dtype=np.float32)

@torch.no_grad()
def rollout_episode(env, pi):
    s, _ = env.reset()
    ep_rews = []
    done = False
    while not done:
        s_t = torch.tensor(s * OBS_SCALE, dtype=torch.float32)
        a, u = pi(s_t)
        s, r, term, trunc, _ = env.step(a.numpy())
        ep_rews.append(r)
        done = term or trunc
    return sum(ep_rews)

def train_one_epoch(env, pi, opt, batch_size=5000, gamma=0.99):

    # empty lists for logging
    batch_states = []
    batch_actions = []
    batch_us = []
    batch_weights = []  # for return weights
    batch_rets = []     # for measuring episode returns
    batch_lens = []     # for measuring episode lengths
    
    # reset episode-specific variables
    s, _ = env.reset()
    ep_rews = []
    done = False
    
    # collect experience by acting in the environment with current policy
    while True:
        s_t = torch.tensor(s * OBS_SCALE, dtype=torch.float32)
        
        # act in environment (no grad during rollout)
        with torch.no_grad():
            a, u = pi(s_t)
        
        s, r, term, trunc, _ = env.step(a.numpy())
        
        # save state, action, reward
        batch_states.append(s_t)
        batch_actions.append(a)
        batch_us.append(u)
        ep_rews.append(r)
        
        done = term or trunc
        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            
            # the weight for each logprob(a_t|s_t) is return from t
            batch_weights += returns(ep_rews, gamma)
            
            # reset episode-specific variables
            s, _ = env.reset()
            ep_rews = []
            done = False
            
            # end experience loop if we have enough of it
            if len(batch_states) > batch_size:
                break
    
    # prepare batch tensors
    states = torch.stack(batch_states)
    actions = torch.stack(batch_actions)
    us = torch.stack(batch_us)
    Gs = torch.tensor(batch_weights, dtype=torch.float32)
    
    # recompute mu, std with gradients (rollout was @no_grad)
    mus = pi.mu(states)
    stds = pi.log_std.exp().expand_as(mus)
    logps = pi.log_prob(actions, us, mus, stds)
    
    # REINFORCE: maximize E[log π * (G - baseline)]
    advantages = Gs - Gs.mean()  # baseline for variance reduction
    loss = -(logps * advantages).mean()
    
    # take a single policy gradient update step
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
    opt.step()
    
    return batch_rets, batch_lens

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous=True)
    env_viz = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    pi = Policy()
    opt = torch.optim.Adam(pi.parameters(), lr=3e-4)
    
    best_reward = -float('inf')
    reward_history = deque(maxlen=100)
    
    if PLOT:
        all_rewards = []
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))
    
    tqdm.write("epoch | r_mean | r_100 | r_best | n_eps")
    tqdm.write("-" * 45)
    
    for i in trange(1000, desc="training", ncols=80, leave=True):
        # run one epoch: collect experience and update policy
        ep_rewards, ep_lens = train_one_epoch(env, pi, opt, batch_size=5000)
        
        reward = np.mean(ep_rewards)
        reward_std = np.std(ep_rewards)
        n_episodes = len(ep_rewards)
        reward_history.append(reward)
        smooth = np.mean(reward_history)
        if PLOT:
            all_rewards.append(reward)
        
        is_best = reward > best_reward
        if is_best:
            best_reward = reward
        
        # print on new best or eval
        if is_best:
            tqdm.write(f"{i:5d} | {reward:6.1f} | {smooth:6.1f} | {best_reward:6.1f} | {n_episodes:4d} 🚀 (±{reward_std:.1f})")
        else:
            tqdm.write(f"{i:5d} | {reward:6.1f} | {smooth:6.1f} | {best_reward:6.1f} | {n_episodes:4d}")
        
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
            eval_reward = rollout_episode(env_viz, pi)
            tqdm.write(f"{'='*45}")
            tqdm.write(f"EVAL: {eval_reward:.1f} | epoch_mean: {reward:.1f}±{reward_std:.1f} ({n_episodes} eps)")
            tqdm.write(f"{'='*45}")
    
    env.close()
    env_viz.close()
    if PLOT:
        plt.ioff()
        plt.show()
    tqdm.write(f"\n✅ done. best: {best_reward:.1f}")
