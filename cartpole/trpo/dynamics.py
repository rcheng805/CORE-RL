import numpy as np

# Baseline with full model of dynamics
def get_full_dynamics(s, u):
    m = 1.
    g = 10.
    l = 1.
    dt = 0.05
    theta = np.arctan(s[1],s[0])
    theta_dot = s[2]
    new_theta = theta + theta_dot*dt + 3*g*np.sin(theta)*(dt^2)/(2*l) + 3*(dt^2)*u/(m*l^2)
    new_theta_dot = theta_dot + 3*g*dt*np.sin(theta)/(2*l) + 3*dt*u/(m*l^2)

    f = np.array([[theta + theta_dot*dt + 3*g*np.sin(theta)*(dt^2)/(2*l)],[theta_dot + 3*g*dt*np.sin(theta)/(2*l)]])
    g = np.array([[3*(dt^2)/(m*l^2)],[3*dt/(m*l^2)]])
    
    return [f,g]
    
# Get linearized dynamics with model error for CartPole problem
def get_linear_dynamics():
    tau = 0.02
    g = 10.
    m = 0.2
    M = 2.2
    l = 0.05
  
    A = np.array([[1., tau, 0., 0.], [0., 1., -tau*m*g*l/(4*M*l/3 - m*l), 0.], [0., 0., 1., tau], [0., 0., M*g*tau/(4*M*l/3 - m*l), 1]])
    B = np.array([[0], [tau/M + tau*m*l/(4*M**2*l/3 - m*M*l)], [0], [-tau/(4*M*l/3 - m*l)]])

    return [A, B]

