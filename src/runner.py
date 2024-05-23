import numpy as np

from rl_lib.algorithms.agent import Agent
from rl_lib.utils.env_wrapper import EnvWrapper

class Runner:
    def __init__(self, env: EnvWrapper, agent: Agent, reward_wrapper=None, seed: int=None):
        self._env = env
        self._env.reset(seed=seed)

        self._agent = agent

        self._step = 0
        self._episode = 0

        self._reward_wrapper = reward_wrapper
    
    def reset(self):
        self._step = 0
        self._episode = 0

    def train_by_episode(self, n_episodes: int, max_step: int, verbose: bool = False):
        max_rewards = -np.inf
        while self._episode < n_episodes:
            state = self._env.reset()

            epi_rewards = 0
            epi_step = 0

            done = False
            while not done and epi_step < max_step:
                action = self._agent.select_action(state)
                next_state, reward, done, _ = self._env.step(action)
                if self._reward_wrapper is not None:
                    custom_reward = self._reward_wrapper(state, action, reward, next_state, done, self._step)
                else:
                    custom_reward = reward
                self._agent.step(state, action, custom_reward, next_state, done, self._step)
                state = next_state
                self._step += 1

                epi_step += 1
                epi_rewards += reward
            
            if verbose:
                if epi_rewards > max_rewards:
                    max_rewards = epi_rewards
                print(f"Episode {self._episode} rewards: {epi_rewards} steps: {epi_step}")
            self._episode += 1

        if verbose:
            print(f"Max rewards: {max_rewards}")
        self.reset()

    def train_by_step(self, n_steps: int, verbose: bool = False):
        max_rewards = -np.inf
        while self._step < n_steps:
            state = self._env.reset()

            epi_rewards = 0
            epi_step = 0

            done = False
            while not done and self._step < n_steps:
                action = self._agent.select_action(state)
                next_state, reward, done, _ = self._env.step(action)
                self._agent.step(state, action, reward, next_state, done, self._step)
                if self._reward_wrapper is not None:
                    custom_reward = self._reward_wrapper(state, action, custom_reward, next_state, done, self._step)
                else:
                    custom_reward = reward
                state = next_state
                self._step += 1

                epi_step += 1
                epi_rewards += reward
            
            if verbose:
                if epi_rewards > max_rewards:
                    max_rewards = epi_rewards
                print(f"Episode {self._episode} rewards: {epi_rewards} steps: {epi_step}")
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

    def visualize(self, env, max_step: int):
        """
        Visualize the agent's performance.

        arguments:
        - env: EnvWrapper, environment -> should be initialized with render_mode="human" like EnvWrapper(gym.make("CartPole-v1", render_mode="human"))
        """
        total_rewards = 0
        epi_step = 0
        self._env = env
        
        state = self._env.reset()
        done = False
        while not done and epi_step < max_step:
            self._env.render()
            action = self._agent.select_action(state, test=True)
            next_state, reward, done, _ = self._env.step(action)

            state = next_state
            total_rewards += reward

            epi_step += 1

            print(f"Step: {epi_step}, rewards: {total_rewards}")
        print(f"Total rewards: {total_rewards}")