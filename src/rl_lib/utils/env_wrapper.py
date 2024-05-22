import numpy as np

class EnvWrapper:
    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, seed=None):
        state = self._env.reset(seed=seed)
        if type(state) != np.ndarray:
            state = np.array(state[0])
        return state
    
    def step(self, action):
        next_state, reward, done, info, _ = self._env.step(action)
        if type(next_state) != np.ndarray:
            next_state = np.array(next_state[0])
        return next_state, reward, done, info
    
    def render(self):
        self._env.render()