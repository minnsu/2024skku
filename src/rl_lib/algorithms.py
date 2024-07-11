from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_lib.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQN:
    def __init__(self, network, optimizer, gamma, batch_size, n_step, buffer_size, update_freq, loss_fn=F.smooth_l1_loss, max_grad_norm=None):
        self.network = network
        self.target_network = deepcopy(self.network)
        self.optimizer = optimizer
        
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_size, self.n_step, gamma, self.batch_size)
        self.n_step = n_step
        self.gamma = gamma
        self.update_freq = update_freq

        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm

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

        with torch.no_grad():
            next_values = self.target_network(next_states)
            next_values = next_values.max(dim=1).values
            target_values = rewards + self.gamma * next_values * (1 - dones)
        
        values = self.network(states)
        loss = self.loss_fn(values[range(self.batch_size), actions], target_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if step % self.update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.item()

class SAC:
    def __init__(self, network, optimizer, gamma, batch_size, buffer_size, update_freq, tau, ent_coef=0.2):
        self.network = network
        self.target_networks = [deepcopy(self.network), deepcopy(self.network)]
        self.optimizer = optimizer
        
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_size, 1, gamma, self.batch_size)
        self.gamma = tau = tau
        self.ent_coef = ent_coef

    def select_action(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        actions, _ = self.network.action_log_prob(states)
        return actions.detach().numpy()
    
    def train(self, step):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        _, log_prob = self.network.action_log_prob(states)
        log_prob = log_prob.reshape(-1, 1)

        with torch.no_grad():
            next_actions, next_log_prob = self.network.action_log_prob(next_states)
            
            next_values_s = [target_network.critic(next_states, next_actions) for target_network in self.target_networks]

            next_values = torch.min(*next_values_s) - self.ent_coef * next_log_prob
            target_values = rewards + self.gamma * next_values * (1 - dones)
        
        values = self.network.critic(states, actions)

        critic_loss = F.mse_loss(values, target_values)
        actor_loss = (self.ent_coef * log_prob - self.network.critic(states, actions)).mean()

        self.optimizer.zero_grad()
        
        critic_loss.backward()
        actor_loss.backward()

        self.optimizer.step()

        if step % self.update_freq == 0:
            # Update target networks using polyak averaging
            for target_network, network in zip(self.target_networks, [self.network, self.network]):
                for target_param, param in zip(target_network.parameters(), network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class PPO:
    def __init__(self):
        pass
