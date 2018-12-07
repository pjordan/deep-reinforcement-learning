import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, config):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(config.seed)
        self.act_fn = config.activation_fn
        self.state_bn = nn.BatchNorm1d(config.state_size)
        self.fc1 = nn.Linear(config.state_size, config.fc1_units)
        self.fc2 = nn.Linear(config.fc1_units, config.fc2_units)
        self.fc3 = nn.Linear(config.fc2_units, config.action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.act_fn(self.fc1(self.state_bn(state)))
        x = self.act_fn(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, config):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(config.seed)
        self.act_fn = config.activation_fn
        self.state_bn = nn.BatchNorm1d(config.state_size)
        self.fcs1 = nn.Linear(config.state_size, config.fcs1_units)
        self.fc2 = nn.Linear(config.fcs1_units+config.action_size, config.fc2_units)
        self.fc3 = nn.Linear(config.fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.act_fn(self.fcs1(self.state_bn(state)))
        x = torch.cat((xs, action), dim=1)
        x = self.act_fn(self.fc2(x))
        return self.fc3(x)
