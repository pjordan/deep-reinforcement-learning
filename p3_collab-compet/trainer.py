import os.path
import pandas as pd
import numpy as np
from tqdm import trange
from unityagents import UnityEnvironment
import agent
import torch
from collections import deque
from tensorboardX import SummaryWriter


def train(config, env, n_episodes=1000, base_port=5005, save_path=None, name=None):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    writer = SummaryWriter(comment=name)
    
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    actor, observer, learner, actor_local, critic_local = agent.create(config, writer=writer)
    
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
                actions = actor.act(states)                        # select an action (for each agent)
                env_info = env.step(actions)[brain_name]           # send all actions to the environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                transition = (states, actions, rewards, next_states, dones) # create transition
                observer.observe(transition)                       # observe
                scores += env_info.rewards                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                if np.any(dones):                                  # exit loop if episode finished
                    break
            learner.learn()
            episode_scores_window.append(np.max(scores))       # save most recent score
            episode_scores.append(np.max(scores))              # save most recent score
            episode_bar.set_postfix(avg_score=np.mean(episode_scores_window))
            writer.add_scalar('data/score.mean', np.mean(scores), episode)
            writer.add_scalar('data/score.max', np.max(scores), episode)

        results = pd.Series(episode_scores, name=name)
        if save_path:
            torch.save(actor_local.state_dict(), os.path.join(save_path,'checkpoint_actor.pth'))
            torch.save(critic_local.state_dict(), os.path.join(save_path, 'checkpoint_critic.pth'))
            results.to_csv(os.path.join(save_path,'results.csv'))    
        writer.close()

