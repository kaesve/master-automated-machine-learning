import numpy as np
import math
import copy
import json

# from domain import *  # Task environments
from .ind_recurrent import RecurrentInd
from utils import rankArray
from .nsga_sort import nsga_sort


class RecurrentNeat():
  """NEAT main class. Evolves population given fitness values of individuals.
  """
  def __init__(self, hyp):
    """Intialize NEAT algorithm with hyperparameters
    Args:
      hyp - (dict) - algorithm hyperparameters

    Attributes:
      p       - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
      pop     - (Ind)      - Current population
      species - (Species)  - Current species   
      innov   - (np_array) - innovation record
                [5 X nUniqueGenes]
                [0,:] == Innovation Number
                [1,:] == Source
                [2,:] == Destination
                [3,:] == New Node?
                [4,:] == Generation evolved
      gen     - (int)      - Current generation
    """
    self.p       = hyp
    self.pop     = [] 
    self.species = [] 
    self.innov   = [] 
    self.gen     = 0  

    self.indType = RecurrentInd

  ''' Subfunctions '''
  from ._variation import evolvePop, recombine
  from ._speciate  import Species, speciate, compatDist,\
                          assignSpecies, assignOffspring  

  def ask(self):
    """Returns newly evolved population
    """
    if len(self.pop) == 0:
      self.initPop()      # Initialize population
    else:
      self.probMoo()      # Rank population according to objectivess
      self.speciate()     # Divide population into species
      self.evolvePop()    # Create child population 

    return self.pop       # Send child population for evaluation

  def tell(self,reward):
    """Assigns fitness to current population

    Args:
      reward - (np_array) - fitness value of each individual
               [nInd X 1]

    """
    for i in range(np.shape(reward)[0]):
      self.pop[i].fitness = reward[i]
      self.pop[i].nConn   = self.pop[i].nConn
  
  def initPop(self):
    """Initialize population with a list of random individuals
    """
    ##  Create base individual
    p = self.p # readability

    nBias = 1
    nInput = p['ann_nInput']
    nOutput = p['ann_nOutput']
    nReccurent = 0
    
    if p['ann_recurrence'] > 0:
      nReccurent = 1
  
    if p['ann_recurrence'] == 1:
      nInput += nReccurent
    
    nTotalIn = nBias + nInput
    nTotalOut = nOutput + nReccurent
    nNode = nTotalIn + nTotalOut
    
    # - Create Nodes -
    nodeId = np.arange(0, nNode, 1)
    node = np.empty((3, nNode), dtype='d')
    node[0,:] = nodeId
    
    # Node types: [1:input, 2:hidden, 3:bias, 4:output]
    node[1,:nBias]             = 4 # Bias
    node[1,nBias:nBias + nInput] = 1 # Input Nodes
    node[1,nBias + nInput:nBias + nInput + nOutput]  = 2 # Output Nodes
    node[1,nBias + nInput + nOutput:nBias + nInput + nOutput + nReccurent]  = 5 # Recurrent Nodes
    
    # Node Activations
    node[2,:] = p['ann_initAct']
    # - Create Conns -
    nConn = nTotalIn * nTotalOut
    ins   = np.arange(0,nTotalIn,1)            # Input and Bias Ids
    outs  = (nTotalIn) + np.arange(0,nTotalOut) # Output Ids
    
    conn = np.empty((5,nConn,), dtype='d')
    conn[0,:] = np.arange(0,nConn,1)    # Connection Id
    # conn[1,:] = np.tile(ins, len(outs)) # Source Nodes
    # conn[2,:] = np.tile(outs,len(ins) ) # Destination Nodes
    conn[3,:] = np.nan                  # Weight Values
    conn[4,:] = 1                       # Enabled?

    for i, inp in enumerate(ins):
      conn[1,i*nTotalOut:(i+1)*nTotalOut] = np.full(nTotalOut, inp) # Source Nodes
      conn[2,i*nTotalOut:(i+1)*nTotalOut] = outs # Destination Nodes
        
    # Create population of individuals with varied weights
    pop = []
    for i in range(p['popSize']):
      _node = node
      _conn = conn
      _nConn = nConn

      _conn[4,:] = np.random.rand(1,_nConn) < p['prob_initEnable']
      if p['ann_recurrence'] == 2:
        recurrenceSources = np.where((_conn[2] == nNode - 1) * _conn[4])[0]
        nRecSrcs = len(recurrenceSources)
        
        recurrenceNodes = np.empty(shape=(3, nRecSrcs))
        recurrenceNodes[0,:] = nNode + np.arange(0, nRecSrcs)
        recurrenceNodes[1,:] = 1
        recurrenceNodes[2,:] = p['ann_initAct']
        _node = np.hstack((node[:,:nBias+nInput], recurrenceNodes, node[:,nBias+nInput:]))

        for i in range(nRecSrcs):

          nRecCons = len(outs)
          recurrenceConns = np.empty((5, nRecCons,), dtype='d')
          recurrenceConns[0,:] = _nConn + np.arange(0,nRecCons,1)    # Connection Id
          recurrenceConns[1,:] = np.full(nRecCons, [recurrenceNodes[0, i]]) # Source Nodes
          recurrenceConns[2,:] = outs # Destination Nodes
          recurrenceConns[3,:] = np.nan                  # Weight Values
          recurrenceConns[4,:] = 0                       # not enabled
          _conn = np.hstack((_conn, recurrenceConns))
          _nConn += nRecCons
      
      
      newInd = self.indType(_conn, _node, recurrence=p['ann_recurrence'])
      newInd.conn[3,:] = (2*(np.random.rand(1,_nConn)-0.5))*p['ann_absWCap']
      newInd.express()
      newInd.birth = 0
      pop.append(copy.deepcopy(newInd))

    # - Create Innovation Record -
    innov = np.zeros([5,pop[0].conn.shape[1]])
    innov[0:3,:] = pop[0].conn[0:3,:]
    innov[3,:] = -1
    
    self.pop = pop
    self.innov = innov

  def probMoo(self):
    """Rank population according to Pareto dominance.
    """
    # Compile objectives
    meanFit = np.asarray([ind.fitness for ind in self.pop])
    nConns  = np.asarray([ind.nConn   for ind in self.pop])
    nConns[nConns==0] = 1 # No connections is pareto optimal but boring...
    objVals = np.c_[meanFit,1/nConns] # Maximize

    # Alternate between two objectives and single objective
    if self.p['alg_probMoo'] < np.random.rand():
      rank = nsga_sort(objVals[:,[0,1]])
    else: # Single objective
      rank = rankArray(-objVals[:,0])

    # Assign ranks
    for i in range(len(self.pop)):
      self.pop[i].rank = rank[i]





