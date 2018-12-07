# Inspired by Shangtong Zhang's Config class
# at https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/config.py

import torch
import torch.nn.functional as F
from rlcc.schedule import ExponentialSchedule, BoundedSchedule

class Config:
   def __init__(self):
      self.device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      #self.num_agents = 2
      #self.in_actor = #
      #self.hidden_in_actor = #
      #self.hidden_out_actor = # 
      #self.out_actor = # 
      #self.in_critic = #
      #self.hidden_in_critic = #
      #self.hidden_out_critic = #
      #self.lr_actor=1.0e-2
      #self.lr_critic=1.0e-2
        
      #self.discount_factor=0.95
      #self.tau=0.02
    
      self.num_agents = 2
      self.state_size = 24
      self.action_size = 2
      self.seed = 0
      self.fc1_units=400
      self.fc2_units=300
      self.fcs1_units=400
      self.activation_fn = F.leaky_relu
      #self.use_same_initial_weights_for_target = False
      self.buffer_size = int(1e5)  # replay buffer size
      self.batch_size = 128        # minibatch size
      self.gamma = 0.99            # discount factor
      self.tau = 1e-3              # for soft update of target parameters
      self.lr_actor = 1e-4         # learning rate of the actor
      self.lr_critic = 1e-3        # learning rate of the critic
      self.weight_decay = 0        # L2 weight decay
    
      self.write_frequency=1000
      self.write_prefix=''
      self.actor_norm_clip=1.0
      self.critic_norm_clip=1.0
      self.noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99), min=0.1)
      self.batches_per_step=10

