import os
import os.path
import numpy as np
import pandas as pd
from trainer import train
from multiprocessing import Pool
from config import Config
import torch.nn.functional as F


base_dir = "saved-models"
base_port = 5015

configs = []
names = []

#
configs.append(Config())
names.append("ddpg-400-300")

# 
configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
names.append("ddpg-200-150")

#
configs.append(Config())
configs[-1].fc1_units = 100
configs[-1].fcs1_units = 100
configs[-1].fc2_units = 75
names.append("ddpg-100-75")

#
configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
configs[-1].activation_fn = F.leaky_relu
names.append("ddpg-200-150-leaky")

#
configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
configs[-1].batch_size = 256
configs[-1].activation_fn = F.leaky_relu
names.append("ddpg-200-150-leaky-bs-256")

def runner(args):
    config, name, port = args
    save_path = os.path.join(base_dir,name)
    os.makedirs(save_path, exist_ok=True)
    train(config, n_episodes=1000, save_path=save_path, base_port=port, name=name)

if __name__ == '__main__':
    for args in  [(c,n,base_port+100*i) for i, (c,n) in enumerate(zip(configs,names))]:
        runner(args)

