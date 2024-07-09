from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_lib.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQN:
    def __init__(self, network, optimizer, gamma, batch_size, n_step, replay_buffer_size, update_freq):
        self.network = network
        self.target_network = deepcopy(self.network)
        self.optimizer = optimizer
        
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.n_step, gamma, self.batch_size)
        self.n_step = n_step
        self.gamma = gamma
        self.update_freq = update_freq

    def select_action(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        values = self.network(states)
        return values.argmax().item()
    
    def train(self, step):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        values = self.network(states)
        next_values = self.target_network(next_states)
        next_values = next_values.max(dim=1).values
        target_values = rewards + self.gamma * next_values * (1 - dones)

        loss = F.mse_loss(values[range(self.batch_size), actions], target_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.item()

class DDPG:
    def __init__(self, network, optimizer, gamma, batch_size, replay_buffer_size, inc_update_factor):
        self.network = network
        self.target_network = deepcopy(self.network)
        self.optimizer = optimizer
        
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(replay_buffer_size, 1, gamma, self.batch_size)
        self.gamma = gamma
        self.inc_update_factor = inc_update_factor

    def select_action(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        action, value = self.network(states)
        return action.item()
    
    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        _, values = self.network(states)
        _, next_values = self.target_network(next_states)
        next_values = next_values.max(dim=1).values
        target_values = rewards + self.gamma * next_values * (1 - dones)

        value_loss = F.mse_loss(values, target_values)
        policy_loss = -self.network(states)[1].mean()

        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_network.load_state_dict(self.inc_update_factor * self.target_network.state_dict() + (1 - self.inc_update_factor) * self.network.state_dict())

        return loss.item()

class SAC:
    def __init__(self):
        pass

class PPO:
    def __init__(self):
        pass
