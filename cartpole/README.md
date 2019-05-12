Code to implement CORE-RL algorithm on CartPole Data. In order to set up the environment, you must replace the "cartpole.py" file in your Gym environment, with the "cartpole.py" environment provided here. This file contains the the modified gym environment, so that the control problem has continuous state-action space. 
All folders contain the following files:
	(1) prior.py - Synthesize control prior.
	(2) dynamics.py - Get (crude) dynamics of the system, for synthesizing the control prior.

For each of the RL algorithms (DDPG, PPO, TRPO), their folder contains the following code,
DDPG folder:
	(1) add_ddpg.py - Run the CORE-RL algorithm with a fixed regularization weight, λ. This can be modified under the comment "Set control prior regularization weight".
	(2) add_ddpg_adaptive.py - Run the CORE-RL algorithm with an adaptive regularization weight, λ. The hyperparameters for the adaptive strategy are tuned through the variables "factor" and "lambda_max".
	(3) replay_buffer_add.py - Create/utilize the replay buffer for the DDPG algorithm.
	
PPO folder:
	(1) ppo.py - Run the CORE-RL algorithm with a fixed regularization weight, λ. This can be modified under the comment "Set control prior regularization weight".
	(2) ppo_adaptive.py - Run the CORE-RL algorithm with an adaptive regularization weight, λ. The hyperparameters for the adaptive strategy are tuned through the variables "factor" and "lambda_max".
	(3) utils.py - Miscellaneous functions for storing the results of the PPO algorithm.

TRPO folder:
	(1) main.py - Run the CORE-RL algorithm.
	(2) learn.py - Class to generate the learning agent using fixed regularization weight, λ. Modify "from trpo_add import TRPO" to "from trpo_adaptive import TRPO" in order to use adaptive regularization weight, λ.
	(3) trpo_add.py - TRPO agent with fixed regularization weight.
	(4) trpo_adaptive.py - TRPO agent with adaptive regularization weight.
	(5) utils.py - Miscellaneous functions to help the agent.
	(6) gae.py - Generalized advantage estimator for estimating the advantage function

