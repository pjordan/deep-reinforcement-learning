# Inspired by Shangtong Zhang's Config class
# at https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/config.py

import torch
import torch.nn.functional as F

class Config:
   def __init__(self):
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.num_agents = 20
      self.state_size = 33
      self.action_size = 4
      self.seed = 0
      self.fc1_units=400
      self.fc2_units=300
      self.fcs1_units=400
      self.activation_fn = F.relu
      self.use_same_initial_weights_for_target = False
      self.buffer_size = int(1e5)  # replay buffer size
      self.batch_size = 128        # minibatch size
      self.gamma = 0.99            # discount factor
      self.tau = 1e-3              # for soft update of target parameters
      self.lr_actor = 1e-4         # learning rate of the actor
      self.lr_critic = 1e-3        # learning rate of the critic
      self.weight_decay = 0        # L2 weight decay
