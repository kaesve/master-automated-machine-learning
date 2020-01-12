
from collections import namedtuple
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import cv2
import math

class LogicGateEnv(gym.Env):

  def __init__(self, gate):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you want to use them (we didn't)
    self.batch   = 500 # Number of images per batch
    self.seed()
    self.viewer = None

    self.state = [
      [ 1, 1 ],
      [ 1, 0 ],
      [ 0, 1 ],
      [ 0, 0 ]
    ]
    gate = gate.lower()
    self.gate = gate
    if gate == "xor":
      self.target = [ 0, 1, 1, 0 ]
    elif gate == "or":
      self.target = [ 1, 1, 1, 0 ]
    elif gate == "nor":
      self.target = [ 0, 0, 0, 1 ]
    elif gate == "xnor":
      self.target = [ 1, 0, 0, 1 ]
    elif gate == "and":
      self.target = [ 1, 0, 0, 0 ]
    elif gate == "nand":
      self.target = [ 0, 1, 1, 1 ]
    else:
      print("uh oh", gate)

    high = np.array(4**2)
    self.action_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1), dtype=np.float32)
    self.observation_space = spaces.Box(-high, high, dtype=np.float32)


  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    ''' Initialize State'''    
    self.t = 0 # timestep
    return self.state
  
  def step(self, action):
    ''' 
    Judge Classification, increment to next batch
    action - [batch x output] - softmax output
    '''
    action = np.ndarray.flatten(action)
    delta = np.abs(self.target - action)
    
    reward = 4 - np.sum(delta)**2
    # print("action", action, "delta", delta, "reward", reward)

    obs = self.state
    return obs, reward, True, {}


# -- Data Sets ----------------------------------------------------------- -- #


Circle = namedtuple("Circle", ["p", "r"])

def circle_sdf(circle, p):
  return np.linalg.norm(circle.p - p) - circle.r


def line(x):
  return 2*x + 3
