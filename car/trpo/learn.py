import tensorflow as tf
import numpy as np
import time, os
import gym
from trpo_add import TRPO
from prior import BasePrior
from scipy.io import savemat
from car_dat import allCars
import datetime

class LEARNER():
        def __init__(self, args, sess):
                self.args = args
                self.sess = sess
                self.prior = BasePrior()
                
                self.env = allCars()
                self.args.max_path_length = self.env.L
                self.agent = TRPO(self.args, self.env, self.sess, self.prior)
                
        def learn(self):
                train_index = 0
                total_episode = 0
                total_steps = 0
                all_logs = list()
                while True:
                        train_index += 1
                        start_time = time.time()
                        train_log = self.agent.train()
                        total_steps += train_log["Total Step"]
                        total_episode += train_log["Num episode"]

                        all_logs.append(train_log)
                        print(train_index)
                        print(train_log["Episode_Avg_diff"])

                        if total_steps > self.args.total_train_step:
                                savemat('data_adaptive_v6_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=all_logs, args=self.args))

                                break
