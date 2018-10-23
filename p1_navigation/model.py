import torch
import torch.nn as nn
import torch.nn.functional as F

RAY_BLOCK_SIZE=5
RAY_BLOCKS=7
VELOCITY_SIZE=2
STATE_SIZE=RAY_BLOCK_SIZE*RAY_BLOCKS+VELOCITY_SIZE
ACTION_SIZE=4

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(STATE_SIZE, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, ACTION_SIZE)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.adv_fc1 = nn.Linear(STATE_SIZE, fc1_units)
        self.adv_fc2 = nn.Linear(fc1_units, fc2_units)
        self.adv_fc3 = nn.Linear(fc2_units, ACTION_SIZE)
        self.v_fc1 = nn.Linear(STATE_SIZE, fc1_units)
        self.v_fc2 = nn.Linear(fc1_units, fc2_units)
        self.v_fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        adv = F.relu(self.adv_fc1(state))
        adv = F.relu(self.adv_fc2(adv))
        adv = self.adv_fc3(adv)

        val = F.relu(self.v_fc1(state))
        val = F.relu(self.v_fc2(val))
        val = self.v_fc3(val)

        return val + adv - adv.mean()

class QNetworkWithRayLayer(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetworkWithRayLayer, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(RAY_BLOCK_SIZE, fc1_units)
        self.fc2 = nn.Linear(RAY_BLOCKS * fc1_units + VELOCITY_SIZE, fc2_units)
        self.fc3 = nn.Linear(fc2_units, ACTION_SIZE)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        chunks = torch.split(state, RAY_BLOCK_SIZE,1)
        transformed_chunks = [F.relu(self.fc1(c)) for c in chunks[:-1]]
        transformed_chunks += [chunks[-1]]
        reduced_state = torch.cat(transformed_chunks,1)

        x = F.relu(self.fc2(reduced_state))
        return self.fc3(x)

class DuelingQNetworkWithRayLayer(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetworkWithRayLayer, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(RAY_BLOCK_SIZE, fc1_units)
        self.adv_fc2 = nn.Linear(RAY_BLOCKS * fc1_units + VELOCITY_SIZE, fc2_units)
        self.adv_fc3 = nn.Linear(fc2_units, ACTION_SIZE)
        self.v_fc2 = nn.Linear(RAY_BLOCKS * fc1_units + VELOCITY_SIZE, fc2_units)
        self.v_fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        chunks = torch.split(state, RAY_BLOCK_SIZE,1)
        transformed_chunks = [F.relu(self.fc1(c)) for c in chunks[:-1]]
        transformed_chunks += [chunks[-1]]
        reduced_state = torch.cat(transformed_chunks,1)

        adv = F.relu(self.adv_fc2(reduced_state))
        adv = self.adv_fc3(adv)

        val = F.relu(self.v_fc2(reduced_state))
        val = self.v_fc3(val)

        return val + adv - adv.mean()
