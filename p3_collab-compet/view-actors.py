from model import Actor, Critic
from config import Config
import torch
from rlcc.act import NetworkActor, StackedActor
from unityagents import UnityEnvironment
import numpy as np
import imageio
import os
from itertools import cycle

configs = []
names = []

# 
configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
names.append("ddpg-400-300-128bs-100p-01s")

configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
names.append("ddpg-400-300-128bs-100p-001s")


configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
names.append("ddpg-400-300-128bs-100p-0001s")

configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
names.append("ddpg-400-300-128bs-10p-01s")

configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
names.append("ddpg-400-300-256bs-100p-001s")

configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
names.append("ddpg-200-150-128bs-100p-001s-3t")

configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
names.append("ddpg-200-150-128bs-100p-001s-4t")

configs.append(Config())
configs[-1].fc1_units = 100
configs[-1].fcs1_units = 100
configs[-1].fc2_units = 75
names.append("ddpg-100-75-128bs-100p-001s-3t")

configs.append(Config())
configs[-1].fc1_units = 100
configs[-1].fcs1_units = 100
configs[-1].fc2_units = 75
names.append("ddpg-100-75-128bs-100p-001s-4t")

configs.append(Config())
configs[-1].fc1_units = 50
configs[-1].fcs1_units = 50
configs[-1].fc2_units = 35
names.append("ddpg-50-35-128bs-100p-001s-3t")

configs.append(Config())
configs[-1].fc1_units = 50
configs[-1].fcs1_units = 50
configs[-1].fc2_units = 35
names.append("ddpg-50-35-128bs-100p-001s-4t")

configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
names.append("ddpg-200-150-256bs-100p-001s-3t")


actors = []
for c,n in zip(configs, names):
    model_path = 'saved-models/{}/checkpoint_actor.pth'.format(n)
    actor_model = Actor(c)
    actor_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    actor_model.to(c.device)
    base_actor = NetworkActor(actor_model, c.device)
    actor = StackedActor([base_actor, base_actor])
    actors.append(actor)

env = UnityEnvironment(file_name="Tennis.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

actor_iter = cycle(actors)

while True:
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations                  
    frames = []
    actor = next(actor_iter)
    while True:
        actions = actor.act(states)                        
        env_info = env.step(actions)[brain_name]
        #    print(env_info.visual_observations)
        #    frames.append(env_info.visual_observations[0])
        states = env_info.vector_observations         
        dones = env_info.local_done                        
        if np.any(dones):                                  
            break

#imageio.mimsave(os.path.join('episode-gifs', 'ddpg-200-150-128bs-100p-001s-3t.gif'), frames, duration=.04)
                    
#env.close()
