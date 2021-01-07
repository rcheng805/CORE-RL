from scipy.io import loadmat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

ppo_0 = loadmat('data0_ppo_v0.mat')
rewards_0 = np.array(ppo_0['data_total'])
ppo_1 = loadmat('data0_ppo_v1.mat')
rewards_1 = np.array(ppo_1['data_total'])
ppo_2 = loadmat('data0_ppo_v2.mat')
rewards_2 = np.array(ppo_2['data_total'])
ppo_3 = loadmat('data0_ppo_v3.mat')
rewards_3 = np.array(ppo_3['data_total'])
ppo_4 = loadmat('data0_ppo_v4.mat')
rewards_4 = np.array(ppo_4['data_total'])
ppo_4 = loadmat('data0_ppo_v4.mat')
rewards_4 = np.array(ppo_4['data_total'])
ppo_5 = loadmat('data0_ppo_v5.mat')
rewards_5 = np.array(ppo_5['data_total'])
ppo_6 = loadmat('data0_ppo_v6.mat')
rewards_6 = np.array(ppo_6['data_total'])
ppo_7 = loadmat('data0_ppo_v7.mat')
rewards_7 = np.array(ppo_7['data_total'])
ppo_8 = loadmat('data0_ppo_v8.mat')
rewards_8 = np.array(ppo_8['data_total'])
ppo_9 = loadmat('data0_ppo_v9.mat')
rewards_9 = np.array(ppo_9['data_total'])

# rewards_all = np.concatenate((rewards_0, rewards_1, rewards_2, rewards_3, rewards_4), axis=0)
rewards_all = np.concatenate((rewards_0, rewards_1, rewards_2, rewards_3, rewards_4, \
                              rewards_5, rewards_6, rewards_7, rewards_8, rewards_9), axis=0)

cmap = matplotlib.cm.get_cmap('jet')
for i in range(10):
    c = cmap(float(i)/10)
    plt.plot(moving_average(rewards_all[i,:], 500), color=c)
plt.xlim([0,25000])
plt.show()


ddpg_0 = loadmat('ddpg/data_prior0_0.mat')
rewards = np.array(ddpg_0['reward'])
rewards_0 = rewards[0,:]
ddpg_1 = loadmat('ddpg/data_prior0_1.mat')
rewards = np.array(ddpg_1['reward'])
rewards_1 = rewards[0,:]
ddpg_2 = loadmat('ddpg/data_prior0_2.mat')
rewards = np.array(ddpg_2['reward'])
rewards_2 = rewards[0,:]
ddpg_3 = loadmat('ddpg/data_prior0_3.mat')
rewards = np.array(ddpg_3['reward'])
rewards_3 = rewards[0,:]
ddpg_4 = loadmat('ddpg/data_prior0_4.mat')
rewards = np.array(ddpg_4['reward'])
rewards_4 = rewards[0,:]
ddpg_5 = loadmat('ddpg/data_prior0_5.mat')
rewards = np.array(ddpg_5['reward'])
rewards_5 = rewards[0,:]
ddpg_6 = loadmat('ddpg/data_prior0_6.mat')
rewards = np.array(ddpg_6['reward'])
rewards_6 = rewards[0,:]
ddpg_7 = loadmat('ddpg/data_prior0_7.mat')
rewards = np.array(ddpg_7['reward'])
rewards_7 = rewards[0,:]
ddpg_8 = loadmat('ddpg/data_prior0_8.mat')
rewards = np.array(ddpg_8['reward'])
rewards_8 = rewards[0,:]
ddpg_9 = loadmat('ddpg/data_prior0_9.mat')
rewards = np.array(ddpg_9['reward'])
rewards_9 = rewards[0,:]

rewards_all = np.stack((rewards_0, rewards_1, rewards_2, rewards_3, rewards_4, \
                rewards_5, rewards_6, rewards_7, rewards_8, rewards_9))

cmap = matplotlib.cm.get_cmap('jet')
for i in range(10):
    c = cmap(float(i)/10)
    plt.plot(moving_average(rewards_all[i,:], 20), color=c)
plt.show()



ddpg_0 = loadmat('ddpg/data/data_prior0_0_20-11-28-07-56.mat')
rewards = np.array(ddpg_0['reward'])
rewards_0 = rewards[0,:]
ddpg_1 = loadmat('ddpg/data/data_prior0_1_20-11-28-08-15.mat')
rewards = np.array(ddpg_1['reward'])
rewards_1 = rewards[0,:]
ddpg_2 = loadmat('ddpg/data/data_prior0_2_20-11-28-08-34.mat')
rewards = np.array(ddpg_2['reward'])
rewards_2 = rewards[0,:]
ddpg_3 = loadmat('ddpg/data/data_prior0_3_20-11-28-08-53.mat')
rewards = np.array(ddpg_3['reward'])
rewards_3 = rewards[0,:]
ddpg_4 = loadmat('ddpg/data/data_prior0_4_20-11-28-09-12.mat')
rewards = np.array(ddpg_4['reward'])
rewards_4 = rewards[0,:]
ddpg_5 = loadmat('ddpg/data/data_prior0_5_20-11-28-09-31.mat')
rewards = np.array(ddpg_5['reward'])
rewards_5 = rewards[0,:]
ddpg_6 = loadmat('ddpg/data/data_prior0_6_20-11-28-09-50.mat')
rewards = np.array(ddpg_6['reward'])
rewards_6 = rewards[0,:]
ddpg_7 = loadmat('ddpg/data/data_prior0_7_20-11-28-10-09.mat')
rewards = np.array(ddpg_7['reward'])
rewards_7 = rewards[0,:]
ddpg_8 = loadmat('ddpg/data/data_prior0_8_20-11-28-10-28.mat')
rewards = np.array(ddpg_8['reward'])
rewards_8 = rewards[0,:]
ddpg_9 = loadmat('ddpg/data/data_prior0_9_20-11-28-10-47.mat')
rewards = np.array(ddpg_9['reward'])
rewards_9 = rewards[0,:]

rewards_all = np.stack((rewards_0, rewards_1, rewards_2, rewards_3, rewards_4, \
                rewards_5, rewards_6, rewards_7, rewards_8, rewards_9))

cmap = matplotlib.cm.get_cmap('jet')
for i in range(10):
    c = cmap(float(i)/10)
    plt.plot(moving_average(rewards_all[i,:], 20), color=c)
plt.show()
