class Agent:
    def __init__(self, network, optimizer, gamma):
        self._network = network
        self._optimizer = optimizer
        self._gamma = gamma
    
    def select_action(self, states):
        raise NotImplementedError
    
    def step(self, states, actions, rewards, next_states, dones, step):
        raise NotImplementedError