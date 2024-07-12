import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetworkTemplate(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(DQNNetworkTemplate, self).__init__()

        self._q_values = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, states):
        return self._q_values(states)

class SACNetworkTemplate(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(SACNetworkTemplate, self).__init__()

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
        action_mean, action_log_std = self._actor(states).chunk(2, dim=-1)
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
    
class PPONetworkTemplate:
    def __init__(self, state_dim, action_dim) -> None:
        self._actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)
        )

        self._critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def action_log_prob(self, states):
        action_mean, action_log_std = self._actor(states).chunk(2, dim=-1)
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