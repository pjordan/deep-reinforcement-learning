from model import Actor, Critic
import rlcc.transitions as transitions
import rlcc.model_triad as triad
from rlcc.noise import ScheduledOUProcess, OUProcess
from rlcc.act import NoisyActor, NetworkActor, StackedActor
from rlcc.observe import BufferedObserver, PreprocessingObserver, StackedObserver
from rlcc.learn import DPG, ReplayLearner, StackedLearner

import torch
import torch.nn.functional as F
import torch.optim as optim

def _make_transition_tensor(transition):
    state, action, reward, next_state, is_terminal = transition
    return transitions.transition(
        torch.tensor(state, dtype=torch.float),
        torch.tensor(action, dtype=torch.float),
        torch.tensor([reward], dtype=torch.float),
        torch.tensor(next_state, dtype=torch.float),
        torch.tensor([is_terminal], dtype=torch.float))

def create(config, writer=None):
    """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
    device = config.device 
    actors = []
    observers = []
    
    
    # Replay memory
    buffer = transitions.buffer(buffer_size=config.buffer_size)
        
    # Actor Network (w/ Target Network)
    actor_local = Actor(config).to(device)
    actor_target = Actor(config).to(device)
    actor_optimizer = optim.Adam(actor_local.parameters(), lr=config.lr_actor)
    actor_triad = triad.bind(actor_local, actor_target, actor_optimizer)
    actor_triad.hard_update()
    
    # Critic Network (w/ Target Network)
    critic_local = Critic(config).to(device)
    critic_target = Critic(config).to(device)
    critic_optimizer = optim.Adam(critic_local.parameters(), lr=config.lr_critic, weight_decay=config.weight_decay)
    critic_triad = triad.bind(critic_local, critic_target, critic_optimizer)
    critic_triad.hard_update()
        
    # Learner
    dpg = DPG(
        actor_triad,
        critic_triad,
        device,
        gamma=config.gamma,
        tau=config.tau,
        writer=writer,
        write_frequency=config.write_frequency,
        actor_norm_clip=config.actor_norm_clip,
        critic_norm_clip=config.critic_norm_clip)
    learner = ReplayLearner(dpg, buffer, device=config.device, batch_size=config.batch_size, batches_per_step=config.batches_per_step)

    base_actor = NetworkActor(actor_local, device)
        
    for i in range(config.num_agents):
        # Actor
        noise_process = ScheduledOUProcess(config.action_size, scale_schedule = config.noise_scale_schedule) # OUProcess(config.action_size)
        actor = NoisyActor(base_actor, noise_process)
        actors.append(actor)
        
        # Observer
        buffered_observer = BufferedObserver(buffer)
        observer = PreprocessingObserver(buffered_observer, _make_transition_tensor)
        observers.append(observer) 
        

    stacked_actor = StackedActor(actors)
    stacked_observers = StackedObserver(observers)
        
    return stacked_actor, stacked_observers, learner, actor_local, critic_local