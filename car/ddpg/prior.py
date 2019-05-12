""" 
Structure for the control prior (Bang-Bang)
"""
import numpy as np
import control
import tensorflow as tf

class BasePrior(object):

    def __init__(self):
        init = 0.

    # Compute control prior action
    def computePrior(self, s):
        s5 = s[12]
        s4 = s[9]
        s3 = s[6]
        s2 = s[3]
        s1 = s[0]
        v5 = s[13]
        v4 = s[10]
        v3 = s[7]
        v2 = s[4]
        v1 = s[1]

        # Preceding Driver
        Kp = 0.4
        Kd = 0.5
        diff_s = (s3 - s4) - (s4 - s5)
        diff_v = (v3 - v4) - (v4 - v5)
        a = Kp*diff_s + Kd*diff_v
        if (a > 0):
            a = 2.5
        elif (a < 0):
            a = -5
        else:
            a = 0.
            
        # Incorporate actuature saturation
        if (a > 3.):
            a = 3.
        if (a < -7):
            a = -7

        return a
