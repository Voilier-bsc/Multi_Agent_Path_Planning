import numpy as np
import torch
import torch.nn.functional as F
import sys
import gym
import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
import random

from skimage.color import rgb2gray
from skimage.transform import resize

import Environment
import os
import A2C

device = 'cpu'

class A2CTestAgent:
    def __init__(self, action_size, state_size, model_path):
        self.action_size = action_size
        self.model = A2C.A2C(action_size, state_size)
        self.model.load_state_dict(torch.load(model_path)['network'])
        self.model.eval()
        

    def get_action(self, state):
        state = np.float32(state / 255.)
        policy, _ = self.model(torch.FloatTensor(state).to(device))
        action = torch.multinomial(policy, num_samples=1).cpu().numpy()
        return action
    
    
model_path = './saved_models/A2C/2022_0621_1709_56/ckpt_80000'
        
Env = Environment.Env()
Env.env_init()
Env.plannar_init()
Env.env_reset()
Env.plot_init()
state_size = Env.state_size
action_size = Env.action_size

a2cAgent = A2CTestAgent(state_size,action_size,model_path)

num_episode = 10


for episode in range(num_episode):
    done = False
    
    score = 0
    Env.env_reset()
    for agent in Env.agent_list:
        agent.state = np.reshape([agent.local_map.cpu().detach().numpy()], (1, Env.local_size, Env.local_size, 5))


    while not done:
        Env.plot()
        for agent in Env.agent_list:
            agent.action = a2cAgent.get_action(agent.state)[0][0]
        
        Env.step()

        for agent in Env.agent_list:
            agent.next_state = np.reshape([agent.local_map.cpu().detach().numpy()], (1, Env.local_size, Env.local_size, 5))
            agent.score += agent.reward
        

        score += Env.agent_list[0].reward

        
        for agent in Env.agent_list:
            agent.state = agent.next_state
    
        ## if done
        Env.done_check()
        if len(Env.agent_list) == 0:
            print("episode: {:3d} | score : {:4.1f}".format(episode, score))
            score = 0
            done = True