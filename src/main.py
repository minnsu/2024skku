import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from runner import Runner
from rl_lib.algorithms.dqn import DQN
from rl_lib.utils.env_wrapper import EnvWrapper

env = EnvWrapper(gym.make("CartPole-v1"))

class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        # ---------------------- DO NOT MODIFY ----------------------
        super(Network, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        # ------------------------------------------------------------

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

network = Network(state_dim, action_dim)
optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
gamma = 0.85
configs = {
    "batch_size": 128,
    "eps": 0.5,
    "eps_decay": 0.99,
    "eps_decay_freq": 100,
    "min_eps": 0.02,
    "n_step": 1,
    "replay_buffer_size": 50000,
    "target_network_update_freq": 300,
    "training_freq": 10,
}

agent = DQN(network, optimizer, gamma, configs)
runner = Runner(env, agent)

runner.train_by_step(50000, verbose=True)

env = EnvWrapper(gym.make("CartPole-v1", render_mode="human"))
runner.visualize(env)