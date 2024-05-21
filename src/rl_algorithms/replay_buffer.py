import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, maxlen, n_step, gamma):
        self._states = deque(maxlen=maxlen)
        self._actions = deque(maxlen=maxlen)
        self._rewards = deque(maxlen=maxlen)
        self._next_states = deque(maxlen=maxlen)
        self._dones = deque(maxlen=maxlen)

        self._n_step = n_step
        self._gamma = gamma
    
    def __len__(self):
        return len(self._states)

    def append(self, state, action, reward, next_state, done):
        self._states.append(state)
        self._actions.append(action)

        for i in range(1, self._n_step):
            if len(self._rewards) < i:
                break
            self._rewards[-i] += reward * (self._gamma ** i)

        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        states = [self._states[i].astype(np.float32) for i in indices]
        actions = [self._actions[i] for i in indices]
        rewards = [self._rewards[i] for i in indices]
        next_states = [self._next_states[i].astype(np.float32) for i in indices]
        dones = [self._dones[i] for i in indices]

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)