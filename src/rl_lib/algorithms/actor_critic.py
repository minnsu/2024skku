import torch
import torch.nn.functional as F
import numpy as np

from rl_lib.algorithms.agent import Agent
from rl_lib.utils.buffer import Buffer

class ActorCritic(Agent):
    """
    Actor-Critic algorithm

    Args:
    actor: torch.nn.Module
    critic: torch.nn.Module
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    gamma: float
    configs: {
        "is_continuous": bool,
        "eps": float,
        "eps_decay": float,
        "eps_decay_freq": int,
        "min_eps": float,
        "n_step": int,
        "entropy_coef": float,
    }
    """
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, gamma, configs):
        self._actor = actor
        self._actor_optimizer = actor_optimizer
        
        self._critic = critic
        self._critic_optimizer = critic_optimizer

        self._gamma = gamma
        self._n_step = configs["n_step"]

        self._is_continuous = configs["is_continuous"]

        self._n_step_buffer = Buffer(self._n_step, 1, self._gamma)

        self._entropy_coef = configs["entropy_coef"]

    def select_action(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        if self._is_continuous:
            mu, sigma = self._actor(states)
            action = torch.normal(mu, sigma)
        else:
            if np.random.rand() < self._eps:
                action = np.random.randint(self._actor._action_dim)
            else:
                values = self._actor(states)
                action = values.argmax().item()
        return action
    
    def step(self, states, actions, rewards, next_states, dones, step):
        self._n_step_buffer.append(states, actions, rewards, next_states, dones)

        if step % self._n_step == 0 and len(self._n_step_buffer) >= self._n_step:
            self.train()
        if not self._is_continuous and step % self._eps_decay_freq == 0:
            self._eps = max(self._min_eps, self._eps * self._eps_decay)
    
    def train(self):
        states, actions, rewards, next_states, dones = self._n_step_buffer.sample_all()

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        returns = torch.zeros_like(rewards)
        R = self._critic(next_states[-1]).detach()
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self._gamma * R * (1 - dones[i])
            returns[i] = R
        
        values = self._critic(states)
        critic_loss = F.mse_loss(values, returns)
        
        advantages = returns - values
        if self._is_continuous:
            mu, sigma = self._actor(states)
            dists = torch.distributions.Normal(mu, sigma)
            log_probs = dists.log_prob(actions)
            actor_loss = -log_probs * advantages - self._entropy_coef * dists.entropy()
        else:
            values = self._actor(states)
            actor_loss = F.cross_entropy(values, actions) * advantages

        self._actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self._actor_optimizer.step()
        
        self._critic_optimizer.zero_grad()
        critic_loss.mean().backward()
        self._critic_optimizer.step()