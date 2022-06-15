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
import Environment

if __name__ == "__main__":
    Env = Environment.Env()
    Env.env_reset()
    Env.plannar_init()
    Env.create_agent()
    Env.plot_init()

    while(True):
        # start = time.time()
        Env.step(Environment.get_action)
        Env.done_check()
        Env.plot()
        if (len(Env.agent_list)==0):
            print("episode end")
            Env.env_reset()
            Env.create_agent()
        # print("inference time : ", time.time() - start)