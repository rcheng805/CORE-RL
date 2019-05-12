Code to implement CORE-RL algorithm on Car-Following dataset. All folders contain the following files:
	(1) car_data_formatted_arc.mat - This mat file contains the measured data from the experimental car-following data, which contains the noisy data of the position, velocity, and acceleration of each car.
	(2) prior.py - Synthesize control prior, which in this case is a bang-bang controller that attempts to keep the car at the midpoint between the cars in front and back. Get the corresponding control prior actions.
	(3) car_dat.py - Code for reading the experimentally gathered data into a simulated environment for the CORE-RL algorithm.

For each of the RL algorithms (DDPG, PPO, TRPO), their folder contains the following code,
# DDPG folder:
	(1) add_ddpg.py - Run the CORE-RL algorithm with a fixed regularization weight, λ. This can be modified under the comment "Set control prior regularization weight". \
	(2) add_ddpg_adaptive.py - Run the CORE-RL algorithm with an adaptive regularization weight, λ. The hyperparameters for the adaptive strategy are tuned through the variables "factor" and "lambda_max". \
	(3) replay_buffer_add.py - Create/utilize the replay buffer for the DDPG algorithm. \
	
# PPO folder:
	(1) ppo.py - Run the CORE-RL algorithm with a fixed regularization weight, λ. This can be modified under the comment "Set control prior regularization weight". \
	(2) ppo_adaptive.py - Run the CORE-RL algorithm with an adaptive regularization weight, λ. The hyperparameters for the adaptive strategy are tuned through the variables "factor" and "lambda_max". \
	(3) utils.py - Miscellaneous functions for storing the results of the PPO algorithm.

# TRPO folder:
	(1) main.py - Run the CORE-RL algorithm. \
	(2) learn.py - Class to generate the learning agent using fixed regularization weight, λ. Modify "from trpo_add import TRPO" to "from trpo_adaptive import TRPO" in order to use adaptive regularization weight, λ. \
	(3) trpo_add.py - TRPO agent with fixed regularization weight. \
	(4) trpo_adaptive.py - TRPO agent with adaptive regularization weight.  \
	(5) utils.py - Miscellaneous functions to help the agent. \
	(6) gae.py - Generalized advantage estimator for estimating the advantage function

