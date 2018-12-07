from model import Actor, Critic
from config import Config
import torch
from rlcc.act import NetworkActor, StackedActor
from unityagents import UnityEnvironment
import numpy as np
import imageio
import os

config = Config()
config.fc1_units = 200
config.fcs1_units = 200
config.fc2_units = 150

model_path = 'saved-models/ddpg-200-150-128bs-100p-001s-3t/checkpoint_actor.pth'

actor_model = Actor(config)
actor_model.load_state_dict(torch.load(model_path))
actor_model.to(config.device)

base_actor = NetworkActor(actor_model, config.device)
actor = StackedActor([base_actor, base_actor])


env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations                  
frames = []

while True:
    actions = actor.act(states)                        
    env_info = env.step(actions)[brain_name]
    frames.append(env_info.visual_observations)
    states = env_info.vector_observations         
    dones = env_info.local_done                        
    if np.any(dones):                                  
        break

imageio.mimsave(os.path.join('episode-gifs', 'ddpg-200-150-128bs-100p-001s-3t.gif'), frames, duration=.04)
                    
env.close()