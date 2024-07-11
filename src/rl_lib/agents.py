import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_lib.algorithms import DQN, DDPG, SAC, PPO

class Agent:
    def __init__(self, env, network, optimizer, gamma):
        self.rewards_list = None
    
    def get_rewards(self):
        return self.rewards_list

class DQNAgent(Agent):
    def __init__(self, env, network, optimizer, gamma, batch_size, n_step, replay_buffer_size, update_freq):
        self.env = env
        self.agent = DQN(network, optimizer, gamma, batch_size, n_step, replay_buffer_size, update_freq)

        self.rewards_list = []

    def train(self, n_episodes, max_steps=300):
        avg_rewards = 0
        for episode in range(1, n_episodes + 1):
            state = self.env.reset()
            done = False

            cur_step = 0
            rewards = 0
            while cur_step <= max_steps and not done:
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

class SACAgent(Agent):
    def __init__(self):
        pass

class PPOAgent(Agent):
    def __init__(self):
        pass
