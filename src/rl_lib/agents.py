import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_lib.algorithms import DQN, SAC, PPO

class Agent:
    def __init__(self, env, network, optimizer, gamma):
        self.rewards_list = None
    
    def get_rewards(self):
        return self.rewards_list
    
    def train(self, n_episodes, max_steps=300):
        raise NotImplementedError
    
    def evaluate(self, n_episodes, max_steps=300):
        raise NotImplementedError

class DQNAgent(Agent):
    def __init__(self, env, network, optimizer, gamma, eps, min_eps, batch_size, n_step, buffer_size, update_freq, loss_fn, max_grad_norm=None):
        self.env = env
        self.agent = DQN(network, optimizer, gamma, batch_size, n_step, buffer_size, update_freq, loss_fn, max_grad_norm)
    
        self.rewards_list = []
        
        self.init_eps = eps
        self.eps = eps
        self.min_eps = min_eps

    def train(self, n_episodes, max_steps=300):
        avg_rewards = 0
        for episode in range(1, n_episodes + 1):
            state = self.env.reset()
            done = False

            cur_step = 0
            rewards = 0
            while cur_step < max_steps and not done:
                if type(state) == tuple:
                    state = state[0]

                if torch.rand(1).item() < self.eps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.replay_buffer.append(state, action, reward, next_state, done)
                state = next_state

                if len(self.agent.replay_buffer) >= self.agent.batch_size:
                    self.agent.train(cur_step)

                rewards += reward
                cur_step += 1

            avg_rewards += rewards
            self.rewards_list.append(rewards)

            self.eps -= (self.init_eps - self.min_eps) / n_episodes
            if episode % 10 == 0:
                print(f"Episode: {episode}, Avg Reward: {avg_rewards / 10}")
                avg_rewards = 0

    def evaluate(self, env):
        rewards = 0
        state = env.reset()
        done = False

        cur_step = 0
        while not done:
            if type(state) == tuple:
                state = state[0]

            action = self.agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            rewards += reward
            cur_step += 1

            print(f"Step: {cur_step}, Reward: {reward}", end="\r")

class SACAgent(Agent):
    def __init__(self, env, network, optimizer, gamma, batch_size, buffer_size, update_freq, tau, ent_coef=0.2):
        self.env = env
        self.agent = SAC(network, optimizer, gamma, batch_size, buffer_size, update_freq, tau, ent_coef)

        self.rewards_list = []

    def train(self, n_episodes, max_steps=300):
        avg_rewards = 0
        for episode in range(1, n_episodes + 1):
            state = self.env.reset()
            done = False

            cur_step = 0
            rewards = 0
            while cur_step < max_steps and not done:
                if type(state) == tuple:
                    state = state[0]

                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.replay_buffer.append(state, action, reward, next_state, done)
                state = next_state

                if len(self.agent.replay_buffer) >= self.agent.batch_size:
                    self.agent.train(cur_step)

                rewards += reward
                cur_step += 1

            avg_rewards += rewards
            self.rewards_list.append(rewards)

            if episode % 10 == 0:
                print(f"Episode: {episode}, Avg Reward: {avg_rewards / 10}")
                avg_rewards = 0

    def evaluate(self, env):
        rewards = 0
        state = env.reset()
        done = False

        cur_step = 0
        while not done:
            if type(state) == tuple:
                state = state[0]

            action = self.agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            rewards += reward
            cur_step += 1

            print(f"Step: {cur_step}, Reward: {reward}", end="\r")

class PPOAgent(Agent):
    def __init__(self, env, network, optimizer, gamma, batch_size, buffer_size, update_freq, clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=None):
        self.env = env
        self.agent = PPO(network, optimizer, gamma, batch_size, buffer_size, update_freq, clip_ratio, ent_coef, vf_coef, max_grad_norm)

        self.rewards_list = []

    def train(self, n_episodes, max_steps=300):
        avg_rewards = 0
        for episode in range(1, n_episodes + 1):
            state = self.env.reset()
            done = False

            cur_step = 0
            rewards = 0
            while cur_step < max_steps and not done:
                if type(state) == tuple:
                    state = state[0]

                action, log_prob = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.replay_buffer.append(state, action, reward, next_state, done, log_prob=log_prob)
                state = next_state

                if len(self.agent.replay_buffer) >= self.agent.batch_size:
                    self.agent.train(cur_step)

                rewards += reward
                cur_step += 1

            avg_rewards += rewards
            self.rewards_list.append(rewards)

            if episode % 10 == 0:
                print(f"Episode: {episode}, Avg Reward: {avg_rewards / 10}")
                avg_rewards = 0

    def evaluate(self, env):
        rewards = 0
        state = env.reset()
        done = False

        cur_step = 0
        while not done:
            if type(state) == tuple:
                state = state[0]

            action = self.agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            rewards += reward
            cur_step += 1

            print(f"Step: {cur_step}, Reward: {reward}", end="\r")