import numpy as np

from rl_lib.algorithms.agent import Agent
from rl_lib.utils.env_wrapper import EnvWrapper

class Runner:
    def __init__(self, env: EnvWrapper, agent: Agent, seed: int=None):
        self._env = env
        self._env.reset(seed=seed)

        self._agent = agent

        self._step = 0
        self._episode = 0
    
    def reset(self):
        self._step = 0
        self._episode = 0

    def train_by_episode(self, n_episodes: int, verbose: bool = False):
        max_rewards = -np.inf
        while self._episode < n_episodes:
            state = self._env.reset()

            epi_rewards = 0            
            done = False
            while not done:
                action = self._agent.select_action(state)
                next_state, reward, done, _ = self._env.step(action)
                self._agent.step(state, action, reward, next_state, done, self._step)
                state = next_state
                self._step += 1

                epi_rewards += reward
            
            if verbose:
                if epi_rewards > max_rewards:
                    max_rewards = epi_rewards
                print(f"Episode {self._episode} rewards: {epi_rewards}")
            self._episode += 1

        if verbose:
            print(f"Max rewards: {max_rewards}")
        self.reset()

    def train_by_step(self, n_steps: int, verbose: bool = False):
        max_rewards = -np.inf
        while self._step < n_steps:
            state = self._env.reset()

            epi_rewards = 0            
            done = False
            while not done and self._step < n_steps:
                action = self._agent.select_action(state)
                next_state, reward, done, _ = self._env.step(action)
                self._agent.step(state, action, reward, next_state, done, self._step)
                state = next_state
                self._step += 1

                epi_rewards += reward
            
            if verbose:
                if epi_rewards > max_rewards:
                    max_rewards = epi_rewards
                print(f"Episode {self._episode} rewards: {epi_rewards}")
            self._episode += 1

        if verbose:
            print(f"Max rewards: {max_rewards}")
        self.reset()

    def test(self, n_episodes: int):
        total_rewards = 0
        max_rewards = -np.inf
        min_rewards = np.inf
        for i in range(n_episodes):
            state = self._env.reset()
            
            epi_rewards = 0
            step = 0
            done = False
            while not done:
                action = self._agent.select_action(state)
                next_state, reward, done, _ = self._env.step(action)
                state = next_state
                
                epi_rewards += reward
                step += 1

            total_rewards += epi_rewards
            max_rewards = max(max_rewards, epi_rewards)
            min_rewards = min(min_rewards, epi_rewards)
            print(f"Episode {i} rewards: {epi_rewards}, steps: {step}")
        print(f"Average rewards: {total_rewards / n_episodes}, Max rewards: {max_rewards}, Min rewards: {min_rewards}")

    def visualize(self, env):
        """
        Visualize the agent's performance.

        arguments:
        - env: EnvWrapper, environment -> should be initialized with render_mode="human" like EnvWrapper(gym.make("CartPole-v1", render_mode="human"))
        """
        total_rewards = 0
        self._env = env
        
        state = self._env.reset()
        done = False
        while not done:
            self._env.render()
            action = self._agent.select_action(state, test=True)
            next_state, reward, done, _ = self._env.step(action)

            state = next_state
            total_rewards += reward

        print(f"Total rewards: {total_rewards}")