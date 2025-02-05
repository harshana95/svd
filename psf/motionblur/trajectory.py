"""
Python version of Kernel trajectory generation from "Modeling Performance of Image Restoration from Motion Blur"
"""

import numpy as np
from numpy.random import uniform, normal
from numpy import sin, cos


def polar(v):
    r = (v[0,0]**2 + v[0,1]**2)**0.5
    return r, np.arctan2(v[0,0], v[0,1])

def cartesian(r, theta):
    vx, vy = r*np.cos(theta), r*np.sin(theta)
    return np.asarray([[vx, vy]])

def create_trajectory(TrajSize, anxiety, N_samples, MaxLength):
    # Default values in original code
    # TrajSize = 64, 
    # anxiety = 0.1*uniform()
    # N_samples = 2000
    # MaxLength = 60

    momentum = 0.7*uniform()   # centripetel in original code
    gaussian = 10*uniform() 
    freqShakes = 0.2*uniform()

    norm_factor = MaxLength/(N_samples-1)
    # Initial angle
    init_angle = 2*np.pi*uniform()
    # Initial velocity vector
    v = cartesian(norm_factor, init_angle)
    if anxiety > 0:
        v =  v*anxiety

    trajectory =  np.zeros([N_samples, 2], dtype=np.float32)
    
    for idx in range(N_samples):
        xt = trajectory[idx:idx+1,:]

        if uniform() < freqShakes*anxiety:
            r, theta = polar(v)
            nextAngle = theta + np.pi + (uniform()-0.5)
            nextDirection = cartesian(2*r, nextAngle)
        else:
            nextDirection = cartesian(0, 0)

        dv = nextDirection + anxiety*( gaussian*normal(0,1,[1,2]) - momentum*xt )*norm_factor
        v = v + dv

        norm_v, _ = polar(v)
        v = v*norm_factor/norm_v

        trajectory[idx+1:idx+2, :] = xt + v

    return trajectory


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    anxiety = 0.005;
    numT = 2000;
    MaxTotalLength = 64;
    np.random.seed(25)

    trajectory = create_trajectory(64, 0.005, 2000, 64)
    plt.scatter(trajectory[:,0], trajectory[:,1])
    plt.show()


