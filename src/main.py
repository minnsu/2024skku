
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_device("cuda")

import gymnasium as gym

np.random.seed(0)
torch.manual_seed(0)

from rl_lib.algorithms import DQN




env = gym.make("CartPole-v1")
network = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
configs = {
    "network": network,
    "lr": 0.001,
    "gamma": 0.95,
    "batch_size": 32,
    "n_step": 1,
    "replay_buffer_size": 10000,
    "update_freq": 100
}
agent = DQN(**configs)

eps = 0.5

rewards = 0
for episode in range(1, 201):
    state = env.reset()
    done = False

    cur_step = 0
    while cur_step <= 300 and not done:
        if type(state) == tuple:
            state = state[0]

        if np.random.rand() < eps:
            action = np.random.randint(2)
        else:
            action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.append(state, action, reward, next_state, done)
        state = next_state

        if len(agent.replay_buffer) >= agent.batch_size:
            agent.train(cur_step)
        
        rewards += reward
        cur_step += 1
    
    if episode % 10 == 0:
        print(f"Episode: {episode}, Avg Reward: {rewards / 10}")
        rewards = 0
        eps = max(0.1, eps * 0.99)
        if np.random.rand() < 0.1:
            eps = 0.5

env.close()
env = gym.make("CartPole-v1", render_mode="human")

state = env.reset()
done = False
cur_step = 0
while not done:
    if type(state) == tuple:
        state = state[0]
    action = agent.select_action(state)
    state, reward, done, _, _ = env.step(action)
    env.render()
    print(cur_step, reward, end="\r")
    cur_step += 1