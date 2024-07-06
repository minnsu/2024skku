from collections import deque

import numpy as np

class ReplayBuffer:
    def __init__(self, maxlen, n_step, gamma, batch_size=32):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)

        self.n_step = n_step
        self.gamma = gamma
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.states)

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)

        for i in range(1, self.n_step):
            if len(self.rewards) < i:
                break
            self.rewards[-i] += reward * (self.gamma ** i)

        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample(self):
        indices = np.random.choice(len(self), self.batch_size, replace=False)
        
        states = [self.states[i].astype(np.float32) for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        next_states = [self.next_states[i].astype(np.float32) for i in indices]
        dones = [self.dones[i] for i in indices]

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample_all(self):
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states), np.array(self.dones)

class PrioritizedReplayBuffer:
    def __init__(self):
        pass