import os.path
import pandas as pd
import numpy as np
from tqdm import trange
from unityagents import UnityEnvironment
from ddpg_agent import Agent
import torch
from collections import deque
from tensorboardX import SummaryWriter

def train(config, n_episodes=1000, base_port=5005, save_path=None, name=None):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    writer = SummaryWriter(comment=name)
    env = UnityEnvironment(
        file_name="Reacher_Linux_NoVis/Reacher.x86_64",
        no_graphics=True,
        base_port=base_port)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    dummy_input = (torch.zeros(1, config.num_agents, config.state_size),)
    agent = Agent(config)
    writer.add_graph(agent.actor_local, dummy_input, True)
    #writer.add_graph(agent.critic_local, dummy_input, True)
    
    num_agents = config.num_agents

    # reset
    env_info = env.reset(train_mode=True)[brain_name]
    
    episode_scores = []                        # list containing scores from each episode
    episode_scores_window = deque(maxlen=100)  # last 100 scores
    
    with trange(n_episodes, desc='episode') as episode_bar:
        for episode in episode_bar:
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations                  # get the current state (for each agent)
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)

            while True:
                actions = agent.act(states)                        # select an action (for each agent)
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                agent.step(states, actions, rewards, next_states, dones, writer=writer) # learn
                scores += env_info.rewards                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                if np.any(dones):                                  # exit loop if episode finished
                    break

            episode_scores_window.append(np.mean(scores))       # save most recent score
            episode_scores.append(np.mean(scores))              # save most recent score
            episode_bar.set_postfix(avg_score=np.mean(episode_scores_window))      
            writer.add_scalar('data/score', np.mean(scores), episode)

        results = pd.Series(episode_scores, name=name)
        if save_path:
            torch.save(agent.actor_local.state_dict(), os.path.join(save_path,'checkpoint_actor.pth'))
            torch.save(agent.critic_local.state_dict(), os.path.join(save_path, 'checkpoint_critic.pth'))
            results.to_csv(os.path.join(save_path,'results.csv'))    
        env.close()
        writer.close()
        return results, agent
