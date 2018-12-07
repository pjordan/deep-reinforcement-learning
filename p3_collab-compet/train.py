import os
import os.path
from trainer import train
from config import Config
from rlcc.schedule import ExponentialSchedule, BoundedSchedule
from unityagents import UnityEnvironment

base_dir = "saved-models"
base_port = 5015

configs = []
names = []

# 
configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
configs[-1].batch_size = 128
configs[-1].tau = 1e-2
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.1)
configs[-1].batches_per_step=100
names.append("ddpg-400-300-128bs-100p-01s")

configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
configs[-1].batch_size = 128
configs[-1].tau = 1e-2
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-400-300-128bs-100p-001s")


configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
configs[-1].batch_size = 128
configs[-1].tau = 1e-2
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.001)
configs[-1].batches_per_step=100
names.append("ddpg-400-300-128bs-100p-0001s")

configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
configs[-1].batch_size = 128
configs[-1].tau = 1e-2
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.1)
configs[-1].batches_per_step=10
names.append("ddpg-400-300-128bs-10p-01s")

configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
configs[-1].batch_size = 128
configs[-1].tau = 1e-2
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.1)
configs[-1].batches_per_step=100
names.append("ddpg-200-150-128bs-100p-01s")

configs.append(Config())
configs[-1].fc1_units = 400
configs[-1].fcs1_units = 400
configs[-1].fc2_units = 300
configs[-1].batch_size = 256
configs[-1].tau = 1e-2
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-400-300-256bs-100p-001s")

configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
configs[-1].batch_size = 128
configs[-1].tau = 1e-3
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-200-150-128bs-100p-001s-3t")

configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
configs[-1].batch_size = 128
configs[-1].tau = 1e-4
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-200-150-128bs-100p-001s-4t")

configs.append(Config())
configs[-1].fc1_units = 100
configs[-1].fcs1_units = 100
configs[-1].fc2_units = 75
configs[-1].batch_size = 128
configs[-1].tau = 1e-3
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-100-75-128bs-100p-001s-3t")

configs.append(Config())
configs[-1].fc1_units = 100
configs[-1].fcs1_units = 100
configs[-1].fc2_units = 75
configs[-1].batch_size = 128
configs[-1].tau = 1e-4
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-100-75-128bs-100p-001s-4t")

configs.append(Config())
configs[-1].fc1_units = 50
configs[-1].fcs1_units = 50
configs[-1].fc2_units = 35
configs[-1].batch_size = 128
configs[-1].tau = 1e-3
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-50-35-128bs-100p-001s-3t")

configs.append(Config())
configs[-1].fc1_units = 50
configs[-1].fcs1_units = 50
configs[-1].fc2_units = 35
configs[-1].batch_size = 128
configs[-1].tau = 1e-4
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-50-35-128bs-100p-001s-4t")

configs.append(Config())
configs[-1].fc1_units = 200
configs[-1].fcs1_units = 200
configs[-1].fc2_units = 150
configs[-1].batch_size = 256
configs[-1].tau = 1e-3
configs[-1].noise_scale_schedule=BoundedSchedule(ExponentialSchedule(1.0, 0.99999), min=0.01)
configs[-1].batches_per_step=100
names.append("ddpg-200-150-256bs-100p-001s-3t")

def runner(env, args):
    config, name, port = args
    save_path = os.path.join(base_dir,name)
    os.makedirs(save_path, exist_ok=True)
    train(config, env, n_episodes=1000, save_path=save_path, base_port=port, name=name)

if __name__ == '__main__':
    env = UnityEnvironment(
        file_name="Tennis_Linux_NoVis/Tennis.x86_64",
        no_graphics=True,
        base_port=base_port)
    
    for args in  [(c,n,base_port+100*i) for i, (c,n) in enumerate(zip(configs,names))]:
        runner(env, args)

    env.close()

