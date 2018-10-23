import pandas as pd
import numpy as np
from tqdm import trange
from unityagents import UnityEnvironment
from dqn_agent import Agent
import torch
from collections import deque


def train(agent_config, n_episodes=2000, max_t=1000, base_port=5005, save_path=None, name=None):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    env = UnityEnvironment(
        file_name="Banana_Linux_NoVis/Banana.x86_64",
        no_graphics=True,
        base_port=base_port)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    eps_start=agent_config.get('eps_start',1.0)
    eps_end=agent_config.get('eps_end',0.01)
    eps_decay=agent_config.get('eps_decay',0.995)
    
    lr = agent_config.get('lr',1e-3)
    lr_decay = agent_config.get('lr_decay',1)
    agent = Agent(seed=0, **agent_config)
    
    # reset
    env_info = env.reset(train_mode=True)[brain_name]
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    with trange(n_episodes, desc='episode') as episode_bar:
        for episode in episode_bar:
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            lr = lr*lr_decay                  # decrease learning rate
            for g in agent.optimizer.param_groups:
                g['lr'] = lr
        
            episode_bar.set_postfix(avg_score=np.mean(scores_window))      
            
        if save_path:
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            
        env.close()
        return pd.Series(scores, name=name)