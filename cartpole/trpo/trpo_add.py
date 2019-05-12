import numpy as np
import tensorflow as tf
import time, os
from utils import *
from gae import GAE
from prior import BasePrior

class TRPO():
    def __init__(self, args, env, sess, prior):
        self.args = args
        self.sess = sess
        self.env = env
        self.prior = prior                
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        print('Observation space', self.observation_space)
        print('Action space', self.action_space)
        # 'Box' observation_space and 'Box' action_space
        self.observation_size = self.env.observation_space.shape[0]
        # np.prod : return the product of array element over a given axis
        self.action_size = self.action_space.shape[0]

        # Build model and create variables
        self.build_model()

    def build_model(self):
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        # Mean of old action distribution
        self.old_action_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.old_action_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])
        '''
        Mean value for each action : each action has gaussian distribution with mean and standard deviation
        With continuous state and action space, use GAUSSIAN DISTRIBUTION, maps  from the input features to the mean of Gaussian distribution for each action
        Seperate set of parameters specifies the log standard deviation of each action
        => The policy is defined by the normnal distribution (mean=NeuralNet(states), stddev= exp(r))
        '''
        self.action_dist_mu, action_dist_logstd = self.build_policy(self.obs)
        # Make log standard shape from [1, action size] => [batch size, action size]
        # tf.tile(A, reps) : construct an tensor by repeating A given by 'reps'
        # Use tf.shape instead of tf.get_shape() when 'None' used in placeholder
        self.action_dist_logstd = tf.tile(action_dist_logstd, (tf.shape(action_dist_logstd)[0], 1))

        # outputs probability of taking 'self.action'
        # new distribution  
        self.log_policy = LOG_POLICY(self.action_dist_mu, self.action_dist_logstd, self.action)
        # old distribution
        self.log_old_policy = LOG_POLICY(self.old_action_dist_mu, self.old_action_dist_logstd, self.action)
        
        # Take exponential to log policy distribution
        '''
        Equation (14) in paper
        Contribution of a single s_n : Expectation over a~q[(new policy / q(is)) * advantace_old]
        sampling distribution q is normally old policy
        '''
        batch_size = tf.shape(self.obs)[0]
        # print('Batch size %d' % batch_size)
        policy_ratio = tf.exp(self.log_policy - self.log_old_policy)
        surr_single_state = -tf.reduce_mean(policy_ratio * self.advantage)
        # tf.shape returns dtype=int32, tensor conversion requested dtype float32
        batch_size = tf.cast(batch_size, tf.float32)
        # Average KL divergence and shannon entropy, averaged over a set of inputs to function mu 
        kl = GAUSS_KL(self.old_action_dist_mu, self.old_action_dist_logstd, self.action_dist_mu, self.action_dist_logstd) / batch_size
        ent = GAUSS_ENTROPY(self.action_dist_mu, self.action_dist_logstd) / batch_size

        self.losses = [surr_single_state, kl, ent]
        #tr_vrbs = tf.trainable_variables()
        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')
        for i in tr_vrbs:
            print(i.op.name)

        '''
            Compute a search direction using a linear approx to objective and quadratic approx to constraint
            => The search direction is computed by approximately solving 'Ax=g' where A is FIM
                Quadratic approximation to KL divergence constraint
        '''
        # Maximize surrogate function over policy parameter 'theta'
        self.pg = FLAT_GRAD(surr_single_state, tr_vrbs)
        # KL divergence where first argument is fixed
        # First argument would be old policy parameters, so keep it constant
        kl_first_fixed = GAUSS_KL_FIRST_FIX(self.action_dist_mu, self.action_dist_logstd) / batch_size
        # Gradient of KL divergence
        first_kl_grads = tf.gradients(kl_first_fixed, tr_vrbs)
        # Vectors we are going to multiply
        self.flat_tangent = tf.placeholder(tf.float32, [None])
        tangent = list()
        start = 0
        for vrbs in tr_vrbs:
            variable_size = np.prod(vrbs.get_shape().as_list())
            param = tf.reshape(self.flat_tangent[start:(start+variable_size)], vrbs.get_shape())
            tangent.append(param)
            start += variable_size
        '''
            Gradient of KL with tangent vector
            gradient_w_tangent : list of KL_prime*y for each variables  
        '''
        gradient_w_tangent = [tf.reduce_sum(kl_g*t) for (kl_g, t) in zip(first_kl_grads, tangent)]
        '''
            From derivative of KL_prime*y : [dKL/dx1, dKL/dx2...]*y
                y -> Ay, A is n by n matrix but hard to implement(numerically solving (n*n)*(n*1))
                so first multiply target 'y' to gradient and take derivation
            'self.FVP'  Returns : [d2KL/dx1dx1+d2KL/dx1dx2..., d2KL/dx1dx2+d2KL/dx2dx2..., ...]*y
            So get (second derivative of KL divergence)*y for each variable => y->JMJy (Fisher Vector Product)
        '''
        self.FVP = FLAT_GRAD(gradient_w_tangent, tr_vrbs)
        # Get actual paramenter value
        self.get_value = GetValue(self.sess, tr_vrbs, name='Policy')
        # To set parameter values
        self.set_value = SetValue(self.sess, tr_vrbs, name='Policy')
        # GAE
        self.gae = GAE(self.sess, self.observation_size, self.args.gamma, self.args.lamda, self.args.vf_constraint)
    
        self.sess.run(tf.global_variables_initializer())        

    def train(self):
        batch_path = self.rollout()
        theta_prev = self.get_value()
        # Get advantage from gae
        advantage_estimated = self.gae.get_advantage(batch_path)

        # Put all paths in batch in a numpy array to feed to network as [batch size, action/observation size]
        # Those batches come from old policy before update theta 
        action_dist_mu = np.squeeze(np.concatenate([each_path["Action_mu"] for each_path in batch_path]))
        action_dist_logstd = np.squeeze(np.concatenate([each_path["Action_logstd"] for each_path in batch_path]))
        observation = np.squeeze(np.concatenate([each_path["Observation"] for each_path in batch_path]))
        action = np.squeeze(np.concatenate([each_path["Action"] for each_path in batch_path]))
        
        feed_dict = {self.obs : observation , self.action : np.expand_dims(np.squeeze(action),1), self.advantage : advantage_estimated, self.old_action_dist_mu : np.expand_dims(np.squeeze(action_dist_mu),1), self.old_action_dist_logstd : np.expand_dims(np.squeeze(action_dist_logstd),1)}
        # Computing fisher vector product : FIM * (policy gradient), y->Ay=JMJy
        def fisher_vector_product(gradient):
            feed_dict[self.flat_tangent] = gradient 
            return self.sess.run(self.FVP, feed_dict=feed_dict)

        policy_g = self.sess.run(self.pg, feed_dict=feed_dict)
        '''
            Linearize to objective function gives : objective_gradient * (theta-theta_old) = g.transpose * delta
            Quadratize to kl constraint : 1/2*(delta_transpose)*FIM*(delta)
            By Lagrangian => FIM*delta = gradient
        '''
        # Solve Ax = g, where A is FIM and g is gradient of policy network parameter
        # Compute a search direction(delta) by conjugate gradient algorithm
        search_direction = CONJUGATE_GRADIENT(fisher_vector_product, -policy_g)

        # KL divergence approximated by 1/2*(delta_transpose)*FIM*(delta)
        # FIM*(delta) can be computed by fisher_vector_product
        # a.dot(b) = a.transpose * b
        kl_approximated = 0.5*search_direction.dot(fisher_vector_product(search_direction))
        # beta
        maximal_step_length = np.sqrt(self.args.kl_constraint / kl_approximated)
        # beta*s
        full_step = maximal_step_length * search_direction

        def surrogate(theta):
            self.set_value(theta)
            return self.sess.run(self.losses[0], feed_dict=feed_dict)

        # Last, we use a line search to ensure improvement of the surrogate objective and sttisfaction of the KL constraint by manually control valud of parameter
        # Start with the maximal step length and exponentially shrink until objective improves
        new_theta = LINE_SEARCH(surrogate, theta_prev, full_step, self.args.num_backtracking, name='Surrogate loss')
        # Update policy parameter theta 
        self.set_value(new_theta, update_info=1)

        # Update value function parameter
        # Policy update is perfomed using the old value function parameter  
        self.gae.train()

        # After update, store values at log
        surrogate_after, kl_after, _ = self.sess.run(self.losses, feed_dict=feed_dict)  
        logs = {"Surrogate loss":surrogate_after, "KL_DIV":kl_after}
        logs["Total Step"] = sum([len(path["Reward"]) for path in batch_path])
        logs["Num episode"] = len([path["Reward"] for path in batch_path])
        logs["Total Sum"] = sum([sum(path["Reward"]) for path in batch_path])
        logs["Diff Sum"] = sum([path["Reward_diff"] for path in batch_path])
        logs["Episode_Avg_reward"] = logs["Total Sum"] / logs["Num episode"]
        logs["Episode_Avg_diff"] = logs["Diff Sum"] / logs["Num episode"]
        return logs


    # Make policy network given states
    def build_policy(self, states, name='Policy'):
        print('Initializing Policy network')
        with tf.variable_scope(name):
            h1 = LINEAR(states, self.args.hidden_size, name='h1')
            h1_nl = tf.nn.relu(h1)
            h2 = LINEAR(h1_nl, self.args.hidden_size, name='h2')
            h2_nl = tf.nn.relu(h2)
            h3 = LINEAR(h2_nl, self.action_size, name='h3')
            # tf.initializer has to be either Tensor object or 'callable' that takes two arguments (shape, dtype)
            init = lambda shape, dtype, partition_info=None : 0.01*np.random.randn(*shape)
            # [1, action size] since it has to be constant through batch axis, log standard deviation
            action_dist_logstd = tf.get_variable('logstd', initializer=init, shape=[1, self.action_size])
        
        return h3, action_dist_logstd
        
    def act(self, obs):
        # Need to expand first dimension(batch axis), make [1, observation size]
        obs_expanded = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs_expanded})
        # Sample from gaussian distribution
        action = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))
        # All shape would be [1, action size]
#       print(action)
        return action, action_dist_mu, action_dist_logstd

    def rollout(self):
        paths = list()
        timesteps = 0
        self.num_epi = 0
        while timesteps < self.args.timesteps_per_batch:
            self.num_epi += 1
            # print('%d episode starts' % self.num_epi)
            obs, action, rewards, done, action_dist_mu, action_dist_logstd, reward_diff = [], [], [], [], [], [], []

            # Baseline reward using only control prior
            s0 = self.env.reset()
            sp = np.copy(s0)
            reward_prior = 0.
            for i in range(self.args.max_path_length):
                a_prior = self.prior.getControl_h(sp)
                sp, reward_p, done_p, _ = self.env.step(a_prior)
                reward_prior += reward_p
                if done_p:
                    break
                
            self.env.reset()
            prev_obs = self.env.unwrapped.reset(s0)
            ep_reward = 0.
            for i in range(self.args.max_path_length):
                # Make 'batch size' axis
                prev_obs = np.squeeze(prev_obs)
                prev_obs_expanded = np.expand_dims(prev_obs, 0)

                # Set regularization weight and get control prior
                lambda_mix = 12.
                a_prior = self.prior.getControl_h(prev_obs)

                #All has shape of [1, action size]
                action_, action_dist_mu_, action_dist_logstd_ = self.act(prev_obs)
                                
                # Mix the actions (RL controller and control prior)
                act = action_/(1+lambda_mix) + (lambda_mix/(1+lambda_mix))*a_prior
                                
                # Take action
                #next_obs, reward_, done_, _ = self.env.step(action_)
                next_obs, reward_, done_, _ = self.env.step(act)
                ep_reward += reward_
                # Store observation                                                               
                obs.append(prev_obs_expanded)
                action.append(action_)
                action_dist_mu.append(action_dist_mu_)
                action_dist_logstd.append(action_dist_logstd_)
                done.append(done_)
                rewards.append(reward_)
                # print(prev_obs, action_, reward_, next_obs, done_)
                prev_obs = next_obs
                if done_:
                    # Make dictionary about path, make each element has shape of [None, observation size/action size]
                    path = {"Observation":np.concatenate(obs),
                    "Action":np.concatenate(action),
                    "Action_mu":np.concatenate(action_dist_mu),
                    "Action_logstd":np.concatenate(action_dist_logstd),
                    # [length,]
                    "Done":np.asarray(done),
                    "Reward":np.asarray(rewards),
                    "Reward_diff":np.asarray(ep_reward - reward_prior)}
                    paths.append(path)
                    # print('%d episode finish at %d steps' % (self.num_epi, i+1))
                    break
            timesteps += len(rewards)
        # print('%d steps collected for batch' % timesteps)
        #print('%d episodes, %d steps is collected for batch' % (self.num_epi, timesteps))
        return paths
        


