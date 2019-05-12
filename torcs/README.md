Installation files and instructions for the TORCS simulator can be found at: <http://torcs.sourceforge.net/index.php?name=Sections&op=viewarticle&artid=3>

# DDPG
(1) ActorNetwork.py - Contains the class for the Actor. \
(2) CriticNetwork.py - Contains the class for the Critic. \
(3) gym_torcs.py - Python wrapper of TORCS to provide an OpenAI-gym-like interface. \
(4) ReplayBuffer.py - Create/utilize the replay buffer for the DDPG algorithm. \
(5) snakeoil3_gym.py - Python script to communicate with TORCS simulator. \
(6) add_ddpg.py - Run the CORE-RL algorithm with a fixed regularization weight, λ. This can be modified under the comment "Set control prior regularization weight". \
(7) add_ddpg_adaptive.py - Run the CORE-RL algorithm with an adaptive regularization weight, λ. The hyperparameters for the adaptive strategy are tuned through the variables "factor" and "lambda_max". \
 
TORCS results are output as .log files because the simulator for distinct training runs must be run on individual sockets.
