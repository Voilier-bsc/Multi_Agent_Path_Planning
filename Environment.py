from asyncio import start_unix_server
import math
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import astar
import copy
import time

device = 'cpu'

c_Blue = torch.tensor((0,0,255)).to(device)         # obastacle
c_Red = torch.tensor((255,0,0)).to(device)          # agent
c_Green = torch.tensor((0,255,0)).to(device)        # target
c_Purple = torch.tensor((255,0,204)).to(device)     # other agent
c_Black = torch.tensor((0,0,0)).to(device)          # waypoint
c_White = torch.tensor((255,255,255)).to(device)

c_Gray_Arr = [180, 160, 140, 120, 100, 80, 60, 40, 20, 0]

class Agent:
    def __init__(self, x, y, target_x, target_y):
        self.x              = x
        self.y              = y
        self.prev_x         = x
        self.prev_y         = x
        self.target_x       = target_x
        self.target_y       = target_y
        self.global_path    = []
        self.reward         = 0
        self.done           = False
        self.id             = -1
        self.local_map      = []
        self.history        = []
        self.action         = -1
        self.state          = []
        self.next_state     = []
        self.score          = 0
        
        

class Env:
    def __init__(self):
        self.grid_size          = 1.0
        self.robot_radius       = 0.5
        
        self.world_x            = 50
        self.world_y            = 50
        
        self.local_size         = 15
        self.side_size          = 10
        self.mask_img           = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 3)) * 255).to(device)
        
        self.num_agent          = 1
        self.agent_list         = []
        self.obs_list           = []
        self.overall_map        = []
        
        self.state_size         = (self.local_size, self.local_size, 5)
        self.action_size        = 9
        
        self.fig = plt.figure()
        plt.subplot(2,2,1)
        self.animation_img_global   = plt.imshow(self.mask_img.cpu().detach().numpy()/255)
        plt.subplot(2,2,2)
        self.animation_img          = plt.imshow(self.mask_img.cpu().detach().numpy()/255)
        plt.subplot(2,2,3)
        self.animation_img_his      = plt.imshow(self.mask_img.cpu().detach().numpy()[:,:,1],cmap='gray',vmin=0, vmax=255)
        plt.subplot(2,2,4)
        self.animation_img_way      = plt.imshow(self.mask_img.cpu().detach().numpy()[:,:,1],cmap='gray',vmin=0, vmax=255)
        
        for i in range(0, self.world_x + 1):
            for j in range(0, self.world_y + 1):
                self.overall_map.append([i,j])
        
        self.complement_map = copy.deepcopy(self.overall_map)
        self.step_count = 0
        
    def env_init(self):
        self.step_count = 0
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

    def env_reset(self):
        self.agent_list = []
        random_position = random.sample(self.complement_map, self.num_agent * 2)
        for i in range(self.num_agent):
            agent = Agent(random_position[i][0],random_position[i][1],
                                random_position[i+self.num_agent][0],random_position[i+self.num_agent][1]) 
            agent.global_path = self.a_star.planning(agent.x, agent.y, agent.target_x, agent.target_y)
            agent.id = i
            self.agent_list.append(agent)
            
        for agent in self.agent_list:
            agent.history.append([agent.x, agent.y])
            agent.local_map = torch.from_numpy(np.ones((self.local_size, self.local_size, 5)) * 255).to(device)
            temp_mask_img = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 3)) * 255).to(device)
            temp_mask_img_his = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 1)) * 255).to(device)
            temp_mask_img_way = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 1)) * 255).to(device)
            
            for obs in self.obs_list: 
                temp_mask_img[obs[0] + self.side_size, obs[1] + self.side_size, :] = c_Blue
                
            for agent_ in self.agent_list:
                if agent.id != agent_.id:
                    temp_mask_img[agent_.x + self.side_size, agent_.y + self.side_size, :] = c_Purple
                    
            temp_mask_img[agent.x + self.side_size, agent.y + self.side_size, :] = c_Red
            temp_mask_img_his[agent.x + self.side_size, agent.y + self.side_size, :] = 0
            temp_mask_img[agent.target_x + self.side_size, agent.target_y + self.side_size, :] = c_Green
            
            for history_xy in reversed(agent.history):
                temp_mask_img_his[history_xy[0] + self.side_size, history_xy[1] + self.side_size, :] = 0
    
            way_ts = np.transpose(agent.global_path)
            diff_way = np.abs(way_ts - np.array([agent.x, agent.y]))
            diff_way_norm = np.linalg.norm(diff_way, axis=1)
            start_id = diff_way_norm.argmin()
            
            current_way = way_ts[:start_id]

            for way_xy in current_way:
                temp_mask_img_way[int(way_xy[0] + self.side_size), int(way_xy[1] + self.side_size), 0] = 0
                
            start_x = int(agent.x + self.side_size-self.local_size/2) + 1
            start_y = int(agent.y + self.side_size-self.local_size/2) + 1

            agent.local_map[:,:,0:3] = temp_mask_img[start_x: start_x+ self.local_size,start_y:start_y+self.local_size,:]
            agent.local_map[:,:,3] = temp_mask_img_his[start_x: start_x+ self.local_size,start_y:start_y+self.local_size,0]
            agent.local_map[:,:,4] = temp_mask_img_way[start_x: start_x+ self.local_size,start_y:start_y+self.local_size,0]

            
        
        
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
            if len(agent.history) > 9:
                agent.history.pop(0)
            agent.history.append([agent.x, agent.y])

            action = agent.action
            # if agent.id == 0:
            #     action = -1
                
            # if agent.id == 1:
            #     action = -1
            
            """
            ---------------
            action function
            ---------------
            9 discrete actions
            """
            
            agent.reward -= 0.01
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
                agent.reward -= 0.01
            
            """
            --------------------
            observation function
            --------------------
            agent position
            obastacle position
            waypoint position
            agent past history
            """
            
            agent.local_map = torch.from_numpy(np.ones((self.local_size, self.local_size, 5)) * 255).to(device)
            temp_mask_img = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 3)) * 255).to(device)
            temp_mask_img_his = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 1)) * 255).to(device)
            temp_mask_img_way = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 1)) * 255).to(device)
            
            for obs in self.obs_list: 
                temp_mask_img[obs[0] + self.side_size, obs[1] + self.side_size, :] = c_Blue
                
            for agent_ in self.agent_list:
                if agent.id != agent_.id:
                    temp_mask_img[agent_.x + self.side_size, agent_.y + self.side_size, :] = c_Purple
                    
            temp_mask_img[agent.x + self.side_size, agent.y + self.side_size, :] = c_Red
            temp_mask_img_his[agent.x + self.side_size, agent.y + self.side_size, :] = 0
            temp_mask_img[agent.target_x + self.side_size, agent.target_y + self.side_size, :] = c_Green
            
            gray_c_arr = c_Gray_Arr[10 - len(agent.history):]
            gray_idx = 0
            for history_xy in agent.history:
                temp_mask_img_his[history_xy[0] + self.side_size, history_xy[1] + self.side_size, :] = gray_c_arr[gray_idx]
                gray_idx += 1
    
            way_ts = np.transpose(agent.global_path)
            diff_way = np.abs(way_ts - np.array([agent.x, agent.y]))
            diff_way_norm = np.linalg.norm(diff_way, axis=1)
            start_id = diff_way_norm.argmin()
            
            current_way = way_ts[:start_id]
            # past_way = way_ts[start_id:]
            
            # for way_xy in past_way:
            #     temp_mask_img_way[int(way_xy[0] + self.side_size), int(way_xy[1] + self.side_size), 0] = 100
                
            for way_xy in current_way:
                temp_mask_img_way[int(way_xy[0] + self.side_size), int(way_xy[1] + self.side_size), 0] = 0
                
            start_x = int(agent.x + self.side_size-self.local_size/2) + 1
            start_y = int(agent.y + self.side_size-self.local_size/2) + 1

            agent.local_map[:,:,0:3] = temp_mask_img[start_x: start_x+ self.local_size,start_y:start_y+self.local_size,:]
            agent.local_map[:,:,3] = temp_mask_img_his[start_x: start_x+ self.local_size,start_y:start_y+self.local_size,0]
            agent.local_map[:,:,4] = temp_mask_img_way[start_x: start_x+ self.local_size,start_y:start_y+self.local_size,0]

                
            """
            ---------------
            reward function
            ---------------
            agent collision reward
            obstacle collision reward
            waypoint reward
            reach goal reward
            """ 
            
            ## agent collision reward
            for agent_ in self.agent_list:
                if((agent.id != agent_.id) and ([agent.x, agent.y] == [agent_.x, agent_.y])):
                    # if agent.id == 0:
                    #     print("agent collision!")
                    agent.reward -= 1
                    agent.done = 1
                    # print("agent collision!")
                    
            ## obstacle collision reward
            if [agent.x, agent.y] in self.occupy_map:
                # if agent.id == 0:
                    # print("obastacle collision!")
                agent.reward -= 1
                agent.done = 1
            
            ## reach goal reward
            if [agent.x, agent.y] == [agent.target_x, agent.target_y]:
                if agent.id == 0:
                    print("goal!")
                agent.reward += 3
                agent.done = 1
            
            ## waypoint reward
            agent.reward -= 0.1 * diff_way_norm.min()
            # if agent.id == 0:
            #     print(agent.reward)
                
        self.step_count += 1


    
    def done_check(self):
        for agent in self.agent_list:
            if agent.done == 1:
                self.agent_list.remove(agent)
                
    def plot_init(self):
        self.fig.canvas.mpl_connect('key_press_event', self.heuristic_move)

    def plot(self):
        self.mask_img = torch.from_numpy(np.ones((self.world_x + self.side_size * 2, self.world_y + self.side_size * 2, 3)) * 255).to(device)
        for obs in self.obs_list: 
            self.mask_img[obs[0] + self.side_size, obs[1] + self.side_size, :] = c_Blue
            
        for agent in self.agent_list:
            for i in range(len(agent.global_path[0])):
                self.mask_img[int(agent.global_path[0][i] + self.side_size), int(agent.global_path[1][i] + self.side_size), :] = c_Black
            self.mask_img[agent.target_x + self.side_size, agent.target_y + self.side_size, :] = c_Green
            self.mask_img[agent.x + self.side_size, agent.y + self.side_size, :] = c_Red
            
            if agent.id == 0:
                local_map = agent.local_map.cpu().detach().numpy()
                self.animation_img.set_data(local_map[:,:,0:3]/255)
                self.animation_img_his.set_data(local_map[:,:,3])
                self.animation_img_way.set_data(local_map[:,:,4])
                
        self.img = self.mask_img.cpu().detach().numpy()
        self.animation_img_global.set_data(self.img/255)
        plt.pause(0.01)

    def append_square_obs(self, center_x, center_y, width, height):
        for i in range(int(center_x - width/2), int(center_x + width/2)):
            for j in range(int(center_y - height/2), int(center_y + height/2)):
                self.obs_list.append([i,j]) 
        
def get_action():
    action = random.randint(0,8)
    return action