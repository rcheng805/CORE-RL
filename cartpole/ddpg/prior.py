""" 
Structure for the control prior (H-infinity)
"""
import numpy as np
import control
import dynamics
import tensorflow as tf

class BasePrior(object):

    def __init__(self, A, B):
        self.state_dim = A.shape[0]
        self.action_dim = B.shape[1]
        Q = np.eye(self.state_dim)
        R = np.eye(self.action_dim)
        [self.P,L,self.K] = control.dare(A,B,Q,R)

    # Get LQR control (not used in this example)
    def getControl(self, state):
        print(self.K)
        action = -np.matmul(self.K, np.squeeze(state))
        return action

    # Get H-infinity controller (synthesized using MATLAB toolbox)
    def getControl_h(self, state):
        K = -np.array([[18.61, 19.68, 37.94, 14.93]])
        action = -np.matmul(self.K, np.squeeze(state))
        return action


