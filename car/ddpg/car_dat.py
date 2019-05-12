"""
Import and process the car data to be fed to the CORE-RL algorithm
"""

import numpy as np
from scipy.io import loadmat
import random
from prior import BasePrior

class allCars():

    # Import data, cut it into episodes, and shuffle
    def __init__(self):
        dat = loadmat('car_data_formatted_arc')
        x = np.copy(np.squeeze(dat['car_dat']))
        aaa = np.arange(len(x))
        random.shuffle(aaa)
        self.data = x[aaa]
        y = np.copy(np.squeeze(dat['car_dat']))
        self.data_orig = y[aaa]
        self.count = 0
        self.episode = -1
        self.L = 100
        self.numCars = 5
        self.dt = 0.1
        self.collision_flag = 0
        
        self.state = np.copy(np.squeeze(self.data[0][0]))
        self.bot_state = np.copy(np.squeeze(self.data[0][0][9:12]))

        self.prior = BasePrior()

    # Get state for the car (headways and velocities)
    def getState(self):
        s = self.getState_arc()
        x = np.copy(s)
        s[0] = 0
        s[3] = x[0] - x[3]
        s[6] = x[3] - x[6]
        s[9] = x[6] - x[9]
        s[12] = x[9] - x[12]
        return s[[6,7,9,10,12,13]]

    def getState_arc(self):
        s = np.copy(np.squeeze(self.data[self.episode][self.count]))
        s[9:12] = np.copy(self.bot_state)
        return s

    # Get the reward for a given state/action
    def getReward(self, action):
        s = self.getState_arc()
        if (action > 0):
            r = -np.abs(s[10])*action
        else:
            r = 0
        if (s[6] - s[9]) < 2:
            r = r - 50.
            if (self.collision_flag == 0):
                print("Collision Penalty")
                print(self.episode)
            self.collision_flag = 1
        if (s[9] - s[12]) < 2:
            r = r - 50.
            if (self.collision_flag == 0):
                print("Collision Penalty")
                print(self.episode)
            self.collision_flag = 1
        if (s[6] - s[9]) < 10:
            r = r - np.abs(100/(s[6] - s[9]))
        if (s[9] - s[12]) < 10:
            r = r - np.abs(100/(s[9] - s[12]))
        return r

    # Reset environment first time (using only control prior)
    def reset_inc(self):
        self.collision_flag = 0
        self.count = 0
        self.episode += 1
        self.state = np.copy(np.squeeze(self.data[self.episode][self.count]))
        self.bot_state = np.copy(np.squeeze(self.data[self.episode][self.count][9:12]))
        return self.getState()

    # Reset environment second time (this time with learning)
    def reset(self):
        self.collision_flag = 0
        self.count = 0
        self.state = np.copy(np.squeeze(self.data_orig[self.episode][self.count]))
        self.bot_state = np.copy(np.squeeze(self.data_orig[self.episode][self.count][9:12]))
        return self.getState()    

    # Simulate next step
    def stepNext(self,a):
        self.dt = 0.1
        x = self.bot_state[0] + self.bot_state[1]*self.dt 
        xdot = self.bot_state[1] + a*self.dt
        xdoubledot = a
        self.bot_state[0] = x
        self.bot_state[1] = xdot
        self.bot_state[2] = xdoubledot
        self.state = self.data[self.episode][self.count]

    # Take next step given action
    def step(self,action):
        #Take action for all cars
        self.count += 1
        s = self.getState()
        if (s[2] <= 6.):
            action = action - np.random.normal(2.0)
        if (s[4] <= 6.):
            action = action + np.random.normal(2.0)
        if (action < -7.):
            action = -7.
        if (action > 3.):
            action = 3.
        
        self.stepNext(action)
        s = self.getState()
        r = self.getReward(action)
        if (self.count == self.L-1):
            self.count = 0
            return s, r, True, action
        else:
            return s, r, False, action

    # Compute control prior action
    def getPrior(self):
        s = self.getState_arc()
        return self.prior.computePrior(s)


