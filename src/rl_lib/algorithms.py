from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_lib.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQN:
    def __init__(self, network, lr, gamma, batch_size, n_step, replay_buffer_size, update_freq):
        self.network = network
        self.target_network = deepcopy(self.network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
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

class RAINBOW:
    def __init__(self, network1, network2, lr, gamma, batch_size, n_step, replay_buffer_size):
        self.network1 = network1
        self.network2 = network2
        self.optimizer1 = optim.Adam(self.network1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.network2.parameters(), lr=lr)

        self.batch_size = batch_size

        self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)
        self.n_step = n_step
        self.gamma = gamma

        # Double DQN, Dueling network, PER, Multi-step, Noisy Net, Categorical DQN
        

class DDPG:
    def __init__(self):
        pass

class SAC:
    def __init__(self):
        pass

class PPO:
    def __init__(self):
        pass
