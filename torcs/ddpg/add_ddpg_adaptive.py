from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json
import logging
import sys
import datetime
import time

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import functools
import copy

class OU(object):
    
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


OU = OU()  # Ornstein-Uhlenbeck Process
MODELS_DIR = "../models/"


def clip_to_range(value, lw=-1, up=1):
    if value > up:
        return up
    if value < lw:
        return lw
    return value


def fold(fun, obs, init):
    return functools.reduce(fun, obs, init)


class Controller():
    def __init__(self, pid_constants=[0, 0, 0], pid_target=0, pid_sensor=0, pid_sub_sensor=0):
        self.pid_constants = pid_constants
        self.pid_target = pid_target
        self.pid_sensor = pid_sensor
        self.pid_sub_sensor = pid_sub_sensor

    def fold_pid(self, acc, lobs):
        return (acc + (self.pid_target - lobs[self.pid_sensor][self.pid_sub_sensor]))

    def pid_execute(self, obs):
        act = self.pid_constants[0] * (
                    self.pid_target - obs[-1][self.pid_sensor][self.pid_sub_sensor]) + \
              self.pid_constants[1] * fold(self.fold_pid, obs, 0) + self.pid_constants[2] * (
                          obs[-2][self.pid_sensor][self.pid_sub_sensor] - obs[-1][self.pid_sensor][
                      self.pid_sub_sensor])
        return act

    def pid_info(self):
        return [self.pid_constants, self.pid_target, self.pid_sensor, self.pid_sub_sensor]


def playGame(amodel, cmodel, train_indicator=0, lambda_mixer=10.0, seeded=1337):  # 1 means Train, 0 means simply Run
    startTime = time.time()
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic
    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 29  # of sensors input

    np.random.seed(seeded)

    vision = False

    EXPLORE = 100000.
    if train_indicator:
        episode_count = 1000
    else:
        episode_count = 2
    max_steps = 10000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    best_distance = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # Now load the weight
    logging.info("Now we load the weight")
    try:
        actor.model.load_weights(amodel)
        critic.model.load_weights(cmodel)
        actor.target_model.load_weights(amodel)
        critic.target_model.load_weights(cmodel)
        logging.info("Weight load successfully")
    except:
        logging.info("Cannot find the weight")
        if train_indicator == 0:
            exit()

    window = 5
    best_para_vals = [2, 0, {'max_params': {'p0': 0.97, 'p1': 0.05, 'p2': 49.98, 'pt': 0}}]
    steer_prog = Controller([best_para_vals[2]['max_params'][i] for i in ['p0', 'p1', 'p2']],
                                   best_para_vals[2]['max_params']['pt'], best_para_vals[0], best_para_vals[1])
    best_para_vals = [5, 0, {'max_params': {'p0': 3.97, 'p1': 0.01, 'p2': 48.79, 'pt': 0.46}}]
    accel_prog = Controller([best_para_vals[2]['max_params'][i] for i in ['p0', 'p1', 'p2']],
                                   best_para_vals[2]['max_params']['pt'], best_para_vals[0], best_para_vals[1])

    lambda_store = np.zeros((max_steps, 1))
    lambda_mix = lambda_mixer
    lambda_max = 40.
    factor = 0.8

    logging.info("TORCS Experiment Start with Lambda,Seed = " + str((lambda_mix,seeded)))
    for i_episode in range(episode_count):
        logging.info("Episode : " + str(i_episode) + " Replay Buffer " + str(buff.count()))
        if np.mod(i_episode, 3) == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack(
            (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

        total_reward = 0.
        tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                   list(ob.wheelSpinVel / 100.0), list(ob.track), [0, 0, 0]]
        window_list = [tempObs[:] for _ in range(window)]


        for j in range(max_steps):
            steer_action = clip_to_range(steer_prog.pid_execute(window_list), -1, 1)
            accel_action = clip_to_range(accel_prog.pid_execute(window_list), 0, 1)
            action_prior = [steer_action, accel_action, 0]

            tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                       list(ob.wheelSpinVel / 100.0), list(ob.track), action_prior]
            window_list.pop(0)
            window_list.append(tempObs[:])

            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            if lambda_mix < 6 and i_episode < 100:
                lambda_mix = 6.

            mixed_act = [a_t[0][k_iter] / (1 + lambda_mix) + (lambda_mix / (1 + lambda_mix)) * action_prior[k_iter] for k_iter in range(3)]

            ob, r_t, done, info = env.step(mixed_act)

            s_t1 = np.hstack(
                (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            # Control prior mixing term
            if j > 0 and i_episode > 50:
                lambda_track = lambda_max * (1 - np.exp(-factor * np.abs(r_t + GAMMA * np.mean(target_q_values[-1] - base_q[-1]))))
                lambda_track = np.squeeze(lambda_track)
            else:
                lambda_track = 10.
            lambda_store[j] = lambda_track
            base_q = copy.deepcopy(target_q_values)

            if np.mod(step, 2000) == 0:
                logging.info("Episode " + str(i_episode) + " Distance " + str(ob.distRaced) + " Lap Times " + str((ob.curLapTime, ob.lastLapTime)))
            step += 1
            if done:
                break
        else:
            env.end()  # This is for shutting down TORCS

        logging.info("#### Episode Reward: " + str(total_reward))
        logging.info("####### Episode Length: " + str(j))
        logging.info("########## Lap Times: " + str((ob.curLapTime, ob.lastLapTime)))
        logging.info("## LastLap: " + str(ob.lastLapTime))
        logging.info("### Lambda: " + str(lambda_mix))
        lambda_mix = np.mean(lambda_store)

        if train_indicator and i_episode > 20:
            totalTime = time.time() - startTime
            if np.mod(i_episode, 3) == 0:
                logging.info("Now we save periodic model at time: " + str((totalTime, totalTime/3600)))
                actor.model.save(MODELS_DIR + "ctrackad_" + str(seeded) + "_actor_model_periodic.h5")
                actor.model.save_weights("ctrackad_" + str(seeded) + "_actor_weights_periodic.h5", overwrite=True)
                with open("ctrackad_" + str(seeded) + "_actor_weights_periodic.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)
                critic.model.save(MODELS_DIR + "ctrackad_" + str(seeded) + "_critic_model_periodic.h5")
                critic.model.save_weights("ctrackad_" + str(seeded) + "_critic_weights_periodic.h5", overwrite=True)
                with open("ctrackad_" + str(seeded) + "_critic_weights_periodic.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        logging.info("TOTAL REWARD @ " + str(i_episode) + "-th Episode  : Reward " + str(total_reward))
        logging.info("Total Step: " + str(j) + "  Distance" + str(ob.distRaced))

    env.end()  # This is for shutting down TORCS
    logging.info("Finish.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--PickTrack', default='gtrack')
    parser.add_argument('--Lambda', default=10)
    parser.add_argument('--Seed', default=1337)
    parser.add_argument('--mode', default=1)  # 0 - Run, 1- Train, 2-Collect observations
    parser.add_argument('--actor', default='a')
    parser.add_argument('--critic', default='c')
    args = parser.parse_args()

    logPath = '../logs'
    logFileName = 'Adap2S' + str(args.Seed) + '_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + args.PickTrack
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, logFileName)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")
    playGame(args.actor, args.critic, train_indicator=args.mode, lambda_mixer=int(args.Lambda), seeded=int(args.Seed))
