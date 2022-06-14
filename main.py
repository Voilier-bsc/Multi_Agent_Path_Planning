import math
from matplotlib import animation
import matplotlib.pyplot as plt
from sympy import Q
import torch
import numpy as np
import random
import astar
import copy
import time


device = 'cuda'

c_Blue = torch.tensor((0,0,255)).to(device)
c_Red = torch.tensor((255,0,0)).to(device)
c_Green = torch.tensor((0,255,0)).to(device)
c_Purple = torch.tensor((255,0,204)).to(device)

c_Black = torch.tensor((0,0,0)).to(device)
c_White = torch.tensor((255,255,255)).to(device)


class Agent:
    def __init__(self, x, y, target_x, target_y):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = x
        self.target_x = target_x
        self.target_y = target_y
        self.global_path = []
        self.reward = 0
        self.done = False
        

class Env:
    def __init__(self):
        self.grid_size = 1.0
        self.robot_radius = 0.5
        self.world_x = 50
        self.world_y = 50
        self.side_size = 10

        self.mask_img      = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 3)) * 255).to(device)
        
        self.num_agent = 100
        self.agent_list = []
        self.obs_list = []
        self.overall_map = []
        
        self.fig = plt.figure()
        self.animation_img = plt.imshow(self.mask_img.cpu().detach().numpy()/255)
        
        for i in range(0, self.world_x + 1):
            for j in range(0, self.world_y + 1):
                self.overall_map.append([i,j])
        
        self.complement_map = copy.deepcopy(self.overall_map)
        
        
    
    def env_reset(self):
        self.obs_list = []
        
        self.append_square_obs(10, 10, 10, 10)
        self.append_square_obs(35, 20, 13, 16)
        self.append_square_obs(17, 30, 4, 8)
        self.append_square_obs(38, 40, 15, 10)
        
        for i in range(0, self.world_x+1):
            self.obs_list.append([i, 0])
            self.obs_list.append([i,self.world_y])
        for i in range(0, self.world_y+1):
            self.obs_list.append([0, i])
            self.obs_list.append([self.world_x,i])


        self.occupy_map = copy.deepcopy(self.obs_list)
        
        for x_y in self.occupy_map:
            try:
                self.complement_map.remove(x_y)
            except:
                continue        

        
    def plannar_init(self):
        self.a_star = astar.AStarPlanner(np.array(self.obs_list)[:,0], np.array(self.obs_list)[:,1], self.grid_size, self.robot_radius)

    def create_agent(self):
        random_position = random.sample(self.complement_map, self.num_agent * 2)
        for i in range(self.num_agent):
            agent = Agent(random_position[i][0],random_position[i][1],
                                random_position[i+self.num_agent][0],random_position[i+self.num_agent][1]) 
            agent.global_path = self.a_star.planning(agent.x, agent.y, agent.target_x, agent.target_y)
            agent.id = i
            self.agent_list.append(agent)
        
    def heuristic_move(self, event):
        if event.key == "up":
            self.agent_list[0].y += 1

        if event.key == "left":
            self.agent_list[0].x -= 1

        if event.key == "down":
            self.agent_list[0].y -= 1

        if event.key == "right":
            self.agent_list[0].x += 1
    
    
    def step(self):

        for agent in self.agent_list:
            action = get_action()
            agent.reward -= 0.1
            if action == 0:
                agent.x += 1
            elif action == 1:
                agent.x += 1
                agent.y -= 1
            elif action == 2:
                agent.y -= 1
            elif action == 3:
                agent.x -= 1
                agent.y -= 1
            elif action == 4:
                agent.x -= 1
            elif action == 5:
                agent.x -= 1
                agent.y += 1
            elif action == 6:
                agent.y += 1
            elif action == 7:
                agent.x += 1
                agent.y += 1
                
            elif action == 8:
                agent.x += 0
                agent.y += 0
                agent.reward -= 0.4
            
            ## collsion
            if [agent.x, agent.y] in self.occupy_map:
                # print("collision!")
                agent.reward -= 5
                agent.done = 1
                
            elif [agent.x, agent.y] == [agent.target_x, agent.target_y]:
                # print("goal!")
                agent.reward += 30
                agent.done = 1
        
            
    
    def done_check(self):
        for agent in self.agent_list:
            if agent.done == 1:
                self.agent_list.remove(agent)
        
        
    def plot_init(self):
        self.animation_img.set_data(self.img/255)
        # self.fig = plt.figure()
        # fig, ax = plt.subplots()
        # fig.canvas.mpl_connect('key_press_event', self.heuristic_move)


    def plot(self):
        self.mask_img = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 3)) * 255).to(device)
        for obs in self.obs_list: 
            self.mask_img[obs[0] + self.side_size, obs[1] + self.side_size, :] = c_Blue
            
        for agent in self.agent_list:
            for i in range(len(agent.global_path[0])):
                self.mask_img[int(agent.global_path[0][i] + self.side_size), int(agent.global_path[1][i] + self.side_size), :] = c_Black
            self.mask_img[agent.target_x + self.side_size, agent.target_y + self.side_size, :] = c_Green
            self.mask_img[agent.x + self.side_size, agent.y + self.side_size, :] = c_Red
        self.img = self.mask_img.cpu().detach().numpy()
        self.animation_img.set_data(self.img/255)
        plt.pause(0.01)

    
    def append_square_obs(self, center_x, center_y, width, height):
        for i in range(int(center_x - width/2), int(center_x + width/2)):
            for j in range(int(center_y - height/2), int(center_y + height/2)):
                self.obs_list.append([i,j]) 
        
    
def get_action():
    action = random.randint(0,8)
    return action





        
Env = Env()
Env.env_reset()
Env.plannar_init()
Env.create_agent()
# Env.plot_init()

while(True):
    start = time.time()
    Env.step()
    Env.done_check()
    Env.plot()
    if (len(Env.agent_list)==0):
        print("episode end")
        Env.env_reset()
        Env.create_agent()
    print("inference time : ", time.time() - start)