import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        # ---------------------- DO NOT MODIFY ----------------------
        super(Network, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        # ------------------------------------------------------------

        pass

    def forward(self, x):
        raise NotImplementedError
    