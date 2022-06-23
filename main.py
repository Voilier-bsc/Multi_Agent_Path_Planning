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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cpu'

date_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S")
save_path = f"./saved_models/A2C/{date_time}"

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.discount_factor = 0.99
        self.learning_rate = 0.00001
        
        self.model = A2C.A2C(self.state_size, self.action_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.writer = SummaryWriter(save_path)
    
    def get_action(self, state):
        state = np.float32(state / 255.)
        policy, _ = self.model(torch.FloatTensor(state).to(device))
        # print(torch.sum(policy))
        # print('policy: ', policy)
        # print(policy.cpu().detach().numpy())

        action = torch.multinomial(policy, num_samples=1).cpu().numpy()
        
        return action
    
    def train_model(self, state, action, reward, next_state, done):
        state = np.float32(state / 255.)
        next_state = np.float32(next_state / 255.)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])
        
        policy, value = self.model(state)
        
        #가치 신경망 (actor)
        with torch.no_grad():
            _, next_value = self.model(next_state)
            target_value = reward + (1-done) * self.discount_factor * next_value
            # print('target: ',target_value)
        critic_loss = F.mse_loss(target_value, value)
        # print('value: ', value)
        # print('next_value', next_value)
        #정책 신경망 (Critic)
        eye = torch.eye(self.action_size).to(device)
        
        one_hot_action = eye[action.view(-1).long()]
        advantage = (target_value - value).detach()
        actor_loss = -(torch.log((one_hot_action * policy).sum(1))*advantage).mean()
        

        total_loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
        
    def save_model(self, num_episode):
        print(f"... Save Model to {save_path}/ckpt_{num_episode} ...")
        torch.save({"network" : self.model.state_dict(),
                    "optimizer" : self.optimizer.state_dict(),
                    }, save_path + '/ckpt_{}'.format(num_episode))
        
    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)
        
        
Env = Environment.Env()
Env.env_init()
Env.plannar_init()
Env.env_reset()
Env.plot_init()
state_size = Env.state_size
action_size = Env.action_size

a2cAgent = A2CAgent(state_size,action_size)

actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0

done = False
num_step = 100000000
Env.env_reset()

save_interval = 100
print_interval = 10

for agent in Env.agent_list:
    agent.state = np.reshape([agent.local_map.cpu().detach().numpy()], (1, Env.local_size, Env.local_size, 5))


for step in range(num_step):
    Env.plot()
    loss_list = []
    
    for agent in Env.agent_list:
        agent.action = a2cAgent.get_action(agent.state)[0][0]
        
    Env.step()
    
    for agent in Env.agent_list:
        agent.next_state = np.reshape([agent.local_map.cpu().detach().numpy()], (1, Env.local_size, Env.local_size, 5))
        agent.score += agent.reward
        
    score += Env.agent_list[0].reward
    ## train 일단 하나의 agent로 진행
    
    for agent in Env.agent_list:

        actor_loss, critic_loss = a2cAgent.train_model(agent.state, [agent.action], [agent.reward], agent.next_state, [agent.done])
    
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        
        agent.state = agent.next_state
    
    ## if done
    Env.done_check()
    if len(Env.agent_list) == 0:
        episode += 1
        scores.append(score)
        score = 0
        
        if episode % print_interval == 0:
            mean_score = np.mean(scores)
            mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
            mean_critic_loss = np.mean(critic_losses)  if len(critic_losses) > 0 else 0
            a2cAgent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step)
            actor_losses, critic_losses, scores = [], [], []
            print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")
        
        # 네트워크 모델 저장 
        if episode % save_interval == 0:
            a2cAgent.save_model(step)
        
        
        Env.env_reset()
        for agent in Env.agent_list:
            agent.state = np.reshape([agent.local_map.cpu().detach().numpy()], (1, Env.local_size, Env.local_size, 5))