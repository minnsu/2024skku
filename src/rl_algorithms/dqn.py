import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from rl_algorithms.agent import Agent
from rl_algorithms.replay_buffer import ReplayBuffer

class DQN(Agent):
    """
    Deep Q-Network algorithm

    Args:
    network: torch.nn.Module
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    gamma: float
    configs: {
        "batch_size": int,
        "eps": float,
        "eps_decay": float,
        "min_eps": float,
        "n_step": int,
        "replay_buffer_size": int,
        "target_network_update_freq": int,
        "training_freq": int,
    }
    """
    def __init__(self, network, optimizer, gamma, configs):
        super().__init__(network, optimizer, gamma)
        
        self._batch_size = configs["batch_size"]
        self._n_step = configs["n_step"]
        
        self._replay_buffer = ReplayBuffer(configs["replay_buffer_size"], self._n_step, self._gamma)
        
        self._target_network_update_freq = configs["target_network_update_freq"]
        self._training_freq = configs["training_freq"]
        
        self._eps = configs["eps"]
        self._eps_decay = configs["eps_decay"]
        self._min_eps = configs["min_eps"]
        
        self._target_network = deepcopy(network)
    
    def select_action(self, states):
        if np.random.rand() < self._eps:
            return np.random.randint(self._network._action_dim)

        states = torch.tensor(states, dtype=torch.float32)
        values = self._network(states)
        return values.argmax().item()

    def step(self, state, action, reward, next_state, done, step):
        self._replay_buffer.append(state, action, reward, next_state, done)
        
        if step % self._training_freq == 0 and len(self._replay_buffer) >= self._batch_size:
            self.train()
            self.eps = max(self._min_eps, self._eps - self._eps_decay)
        if step % self._target_network_update_freq == 0:
            del self._target_network
            self._target_network = deepcopy(self._network)
    
    def train(self):
        states, actions, rewards, next_states, dones = self._replay_buffer.sample(self._batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.int32)

        q_values = self._network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self._target_network(next_states).max(dim=1).values
        target_values = rewards + self._gamma * next_q_values * (1 - dones)

        self._optimizer.zero_grad()

        loss = F.mse_loss(q_values, target_values)
        loss.backward()

        self._optimizer.step()
