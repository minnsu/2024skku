
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_device("cuda")

import gymnasium as gym

np.random.seed(0)
torch.manual_seed(0)

from rl_lib.agents import DQNAgent, SACAgent, PPOAgent


env = gym.make("MountainCarContinuous-v0")

class PPONetworkTemplate(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(PPONetworkTemplate, self).__init__()

        self._actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)
        )

        self._critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def action_log_prob(self, states):
        if states.shape == (2, 1):
            states = states.reshape(2)
        action_mean, action_log_std = self._actor(states).chunk(2, dim=-1)
        action_log_std = torch.clamp(action_log_std, -5, 0.5)
        action_std = torch.exp(action_log_std)
        
        # using reparametrization trick
        action = action_mean + action_std * torch.randn_like(action_mean)
        action_log_prob = torch.distributions.Normal(action_mean, action_std).log_prob(action).sum(dim=-1)

        return action, action_log_prob
    
    def actor(self, states):
        return self.action_log_prob(states)[0]

    def critic(self, states, actions):
        return self._critic(torch.cat([states, actions], dim=-1))

    def forward(self, states):
        return self.action_log_prob(states), self.critic(states)
    
network = PPONetworkTemplate(env.observation_space.shape[0], env.action_space.shape[0])
configs = {
    "env": env,
    "network": network,
    "optimizer": torch.optim.Adam(network.parameters(), lr=1e-3),
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 10000,
    "update_freq": 100,
    "ent_coef": 0.2,
    "vf_coef": 0.5,
    "clip_ratio": 0.05,
}

agent = PPOAgent(**configs)
agent.train(200, max_steps=1000)
env.close()

env = gym.make("MountainCarContinuous-v0", render_mode="human")
agent.evaluate(env)