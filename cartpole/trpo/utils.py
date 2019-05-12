import tensorflow as tf
import numpy as np

# Flatten gradient along all variables
def FLAT_GRAD(loss, vrbs):
        # tf.gradients returns list of gradients w.r.t variables
        '''
        If 'loss' argument is list, tf.gradients returns sum of gradient of each loss element for each variables 
        tf.gradients([y,z]) => [dy/dx+dz/dx]
        '''
        grads = tf.gradients(loss, vrbs)
        # Returns gradient of each variable element
        # Each gradient has same shape with variable
        #return tf.concat(0, [tf.reshape(g, [np.prod(v.get_shape().as_list()),]) for (g, v) in zip(grads, vrbs)])
        return tf.concat(values=[tf.reshape(g, [np.prod(v.get_shape().as_list()),]) for (g,v) in zip(grads,vrbs)], axis=0)

# y -> Hy
def HESSIAN_VECTOR_PRODUCT(func, vrbs, y):
	first_derivative = tf.gradients(func, vrbs)
	flat_y = list()
	start = 0
	for var in vrbs:
		variable_size = np.prod(var.get_shape().as_list())
		param = tf.reshape(y[start:(start+variable_size)], var.get_shape())
		flat_y.append(param)
		start += variable_size
	# First derivative * y
	gradient_with_y = [tf.reduce_sum(f_d * f_y) for (f_d, f_y) in zip(first_derivative, flat_y)]
	HVP = FLAT_GRAD(gradient_with_y, vrbs)
	return HVP 


# mu1, logstd1 : [batch size, action size]
def LOG_POLICY(mu, logstd, action):
	# logstd : log(standard_deviation)
	# variance : exponential(2*log(std))
	variance = tf.exp(2*logstd)
	# Take log to gaussian formula
	log_prob = -tf.square(action - mu) / (2*variance) - 0.5*tf.log(2*np.pi) - logstd
	# Make [batch size, ] sum along 'action size' axis
	# Doing sum becuase it is log scale => actually it is product of probability of each action index
	return tf.reduce_sum(log_prob, 1)


# All argument : [batch size, action size]
# KL divergence between parameterized Gaussian
'''
	P ~ N(mu1, sig1), Q ~ N(mu2, sig2)
	KL(p,q) = log(sig2/sig1) + (sig1**2 + (mu1-mu2)**2)/2(sig2**2) - 0.5
	Referenced at : https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussian
'''
def GAUSS_KL(mu1, logstd1, mu2, logstd2):
	variance1 = tf.exp(2*logstd1)
	variance2 = tf.exp(2*logstd2)

	kl = logstd2 - logstd1 + (variance1 + tf.square(mu1 - mu2))/(2*variance2) - 0.5
	return tf.reduce_sum(kl)
	
'''
	Entropy of Gaussian : Expectation[-log(p(x))]
	integral[p(x) *(-logp(x))] : (1+log(2pi(sig**2)))/2
'''
def GAUSS_ENTROPY(mu, logstd):
	variance = tf.exp(2*logstd)
	
	entropy = (1 + tf.log(2*np.pi*variance))/2
	return tf.reduce_sum(entropy)

def GAUSS_KL_FIRST_FIX(mu, logstd):
	# First argument is old policy, so keep it unchanged through tf.stop_gradient 
	mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
	mu2, logstd2 = mu, logstd
	return GAUSS_KL(mu1, logstd1, mu2, logstd2)

'''
	Conjugate gradient : used to calculate search direction
	Find basis which satisfies <u,v>=u.transpose*Q*v = 0(Q-orthogonal and hessian of objective function)
	Assumed that Q is symmetry and positive semidefinite matrix(Like hessian)
	Numerical solving Qx=b, Here Q is FIM => solving Ax=g
'''
def CONJUGATE_GRADIENT(fvp, y, k=10, tolerance=1e-6):
	# Given intial guess, r0 := y-fvp(x0), but our initial value is x0 := 0 so r0 := y
	p = y.copy()
	r = y.copy()	
	x = np.zeros_like(y)
	r_transpose_r = r.dot(r)
	for i in range(k):
		FIM_p = fvp(p)
		# alpha := r.t*r/p.t*A*p
		alpha_k = r_transpose_r / p.dot(FIM_p)
		#x_k+1 := x_k + alpha_k*p
		x += alpha_k*p
		#r_k+1 := r_k - alpha_k*A*p
		r -= alpha_k*FIM_p
		# beta_k = r_k+1.t*r_k+1/r_k.t*r_k
		new_r_transpose_r = r.dot(r)
		beta_k = new_r_transpose_r / r_transpose_r
		# p_k+1 := r_k+1 + beta_k*p_k
		p = r + beta_k*p
		r_transpose_r = new_r_transpose_r
		if r_transpose_r < tolerance:
			break
	return x



def LINE_SEARCH(surr, theta_prev, full_step, num_backtracking=10, name=None):
	prev_sur_objective = surr(theta_prev)
	# backtracking :1,1/2,1/4,1/8...
	for num_bt, fraction in enumerate(0.5**np.arange(num_backtracking)):
		# Exponentially shrink beta
		step_frac = full_step*fraction
		# theta -> theta + step
		theta_new = theta_prev + step_frac
		new_sur_objective = surr(theta_new)
		# '-' surrogate loss should be minimized
		sur_improvement = prev_sur_objective - new_sur_objective
		if sur_improvement > 0:
			#print('%s improved from %3.4f to %3.4f' % (name, prev_sur_objective, new_sur_objective))
			return theta_new
	print('Objective not improved')	
	return theta_prev
			

def LINEAR(x, hidden, name=None):
	with tf.variable_scope(name or 'L'):
		weight = tf.get_variable('Weight', [x.get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.05))
		bias = tf.get_variable('Bias', [hidden,], initializer=tf.constant_initializer(0))
		weighted_sum = tf.matmul(x, weight) + bias
	return weighted_sum


'''
	'x' should be array has shape of [batch size,]
	Ex ) x = [x1,x2,x3,x4...]
	Return : [x1+df*x2+(df**2)*x3..., x2+df*x3+(df**2)*x4....., ...]
'''
def DISCOUNT_SUM(x, discount_factor, print_info=None):
	size = x.shape[0]
	if print_info is not None:
		print('Input shape', size, 'Discount_factor', discount_factor)
	discount_sum = np.zeros((size,))
	# x[::-1] is reverse of x
	for idx, value in enumerate(x[::-1]):
		discount_sum[:size-idx] += value
		if size-idx-1 == 0:
			break
		discount_sum[:size-idx-1] *= discount_factor

	return discount_sum


# Get actual value
class GetValue:
	def __init__(self, sess, variable_list, name=None):
		self.name = name
		self.sess = sess
		self.op_list = tf.concat(axis=0, values=[tf.reshape(v, [np.prod(v.get_shape().as_list())]) for v in variable_list])

	# Use class instance as function
	def __call__(self):
		#print('Getting %s parameter value' % self.name)
		return self.op_list.eval(session=self.sess)

# Set parameter value
class SetValue:
	def __init__(self, sess, variable_list, name=None):
		self.name = name
		self.sess = sess
		shape_list = list()
		for i in variable_list:
			shape_list.append(i.get_shape().as_list())
		total_variable_size = np.sum(np.prod(shapes) for shapes in shape_list)
		print('Total variable size : %d' % total_variable_size)
		self.var_list = var_list = tf.placeholder(tf.float32, [total_variable_size])
		start = 0
		assign_ops = list()
		for (shape, var) in zip(shape_list, variable_list):
			variable_size = np.prod(shape)
			assign_ops.append(tf.assign(var, tf.reshape(var_list[start:(start+variable_size)], shape)))
			start += variable_size
		# Need '*' to represenet list
		self.op_list = tf.group(*assign_ops)
			
	def __call__(self, var, update_info=0):
		#if update_info:
		#	print('Update %s parameter' % self.name)
		self.sess.run(self.op_list, feed_dict={self.var_list:var})


if __name__ == "__main__":
#	a = np.array([1,2,3])
#	b = DISCOUNT_SUM(a, 0.5)
#	print(b.shape)
#	print(b)
	x = tf.Variable(np.random.randn(3,4))
	y = tf.Variable(np.random.randn(3,5))
	f = tf.pow(x, 2) + 2*y + tf.pow(y, 2) + 4*x
	r = COMPUTE_HESSIAN(f, [x,y])
	print(r)



