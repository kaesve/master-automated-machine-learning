
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

class RegressionEnv(gym.Env):

  def __init__(self):
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

    self.circle = Circle(p=np.array([ 0, 0 ]), r=1)

    self.trainSet = None
    self.target   = None

    high = np.array([np.inf]*4)
    self.action_space  = spaces.Box(low=-10, high=10, shape=(1, 1), dtype=np.float32)
    self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    self.state = None
    self.trainOrder = None
    self.currIndx = None

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def generate_state(self):
    self.state = 3*np.random.normal(size=(self.batch, 2))
    
    self.target = [ circle_sdf(self.circle, row) for row in self.state ]

  def reset(self):
    ''' Initialize State'''    
    #print('Lucky number', np.random.randint(10)) # same randomness?
    self.t = 0 # timestep
    # self.seed()

    self.generate_state()

    return self.state
  
  def step(self, action):
    ''' 
    Judge Classification, increment to next batch
    action - [batch x output] - softmax output
    '''
    action = np.ndarray.flatten(action)
    delta = self.target - action
    # print("example: %.4f, (%.4f, %.4f), %.4f" % (action[0], self.state[0][0], self.state[0][1], self.target[0]))
    # print(self.state)
    mse = np.dot(delta, delta)/len(action)
    reward = -mse

    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t += 1
      done = False
      if self.t >= self.t_limit:
        done = True

      self.generate_state()
    else:
      done = True

    obs = self.state
    return obs, reward, done, {}


# -- Data Sets ----------------------------------------------------------- -- #


Circle = namedtuple("Circle", ["p", "r"])

def circle_sdf(circle, p):
  return np.linalg.norm(circle.p - p) - circle.r


def line(x):
  return 2*x + 3
