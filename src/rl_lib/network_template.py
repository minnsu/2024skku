import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetworkTemplate(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(DQNNetworkTemplate, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, states):
        return self.net(states)

class DDPGNetworkTemplate(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(DDPGNetworkTemplate, self).__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.critic_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def actor(self, states):
        return self.actor_net(states)
    
    def critic(self, states, actions):
        return self.critic_net(torch.cat([states, actions], dim=1))
    
    def forward(self, states):
        action = self.actor(states)
        q_value = self.critic(states, action)
        return action, q_value
