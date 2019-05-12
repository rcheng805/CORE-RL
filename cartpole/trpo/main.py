import tensorflow as tf
import argparse
import numpy as np
import os
from learn import LEARNER


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gamma', type=float, default=0.995, help='Importantly determines the scale of the value function, introduces bias to policy gradient regrardeless of value function')
	# If lambda is 1, it is unbiased generalized advantage estimator
	# gamma-just
	parser.add_argument('--lamda', type=float, default=0.98, help='Best lambda value is lower than gamma, empirically lambda introduces far less bias than gamma for a reasonably accruate value function')
	parser.add_argument('--kl_constraint', type=float, default=0.006)
	parser.add_argument('--num_backtracking', type=int, default=18)
	parser.add_argument('--hidden_size', type=int, default=32)
	#parser.add_argument('--monitor', type=str2bool, default='n')
	parser.add_argument('--timesteps_per_batch', type=int, default=4e3)	
	parser.add_argument('--vf_constraint', type=float, default=1e-2)
	#parser.add_argument('--save_interval', type=int, default=20)
	parser.add_argument('--total_train_step', type=int, default=4e6)
	#parser.add_argument('--log_dir', type=str, default='./logs')
	#parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
	parser.add_argument('--env_name', type=str, default='CartPole-v1')
	parser.add_argument('--is_train', type=str2bool, default='t')
		
	args = parser.parse_args()
	print(args)
    
	#if not os.path.exists(args.checkpoint_dir):
	#	os.mkdir(args.checkpoint_dir)
	#if not os.path.exists(args.log_dir):
	#	os.mkdir(args.log_dir)
        
	config = tf.ConfigProto()
	config.log_device_placement = False
	config.gpu_options.allow_growth = True
		
	with tf.Session(config=config) as sess:
		# Make TRPO_GAE Learner
		trpo_gae_learner = LEARNER(args, sess) 	
		if args.is_train:
			trpo_gae_learner.learn()
		else:
			pass


def str2bool(v):
	if v.lower() in ('yes', 'y', 't', 'true'):
		return True
	else:
		return False



if __name__ == "__main__":
	main()
