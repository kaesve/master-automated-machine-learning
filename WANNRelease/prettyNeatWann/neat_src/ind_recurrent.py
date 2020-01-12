import numpy as np
import copy
from .ann import getLayer, getNodeOrder, applyAct


class RecurrentInd():
  """Individual class: genes, network, and fitness
  """ 
  def __init__(self, conn, node, recurrence=0):
    """Intialize individual with given genes
    Args:
      conn - [5 X nUniqueGenes]
             [0,:] == Innovation Number
             [1,:] == Source
             [2,:] == Destination
             [3,:] == Weight
             [4,:] == Enabled?
      node - [3 X nUniqueGenes]
             [0,:] == Node Id
             [1,:] == Type (1=input, 2=output, 3=hidden, 4=bias, 5=recurrent)
             [2,:] == Activation function (as int)
  
    Attributes:
      node    - (np_array) - node genes (see args)
      conn    - (np_array) - conn genes (see args)
      nInput  - (int)      - number of inputs
      nOutput - (int)      - number of outputs
      wMat    - (np_array) - weight matrix, one row and column for each node
                [N X N]    - rows: connection from; cols: connection to
      wVec    - (np_array) - wMat as a flattened vector
                [N**2 X 1]    
      aVec    - (np_array) - activation function of each node (as int)
                [N X 1]    
      nConn   - (int)      - number of connections
      fitness - (double)   - fitness averaged over all trials (higher better)
      X fitMax  - (double)   - best fitness over all trials (higher better)
      rank    - (int)      - rank in population (lower better)
      birth   - (int)      - generation born
      species - (int)      - ID of species
    """
    self.node    = np.copy(node)
    self.conn    = np.copy(conn)
    self.nInput  = sum(node[1,:]==1)
    self.nOutput = sum(node[1,:]==2)
    self.nRecurrent = sum(node[1,:]==5)
    self.wMat    = []
    self.wVec    = []
    self.aVec    = []
    self.nConn   = np.sum(conn[4])
    self.fitness = [] # Mean fitness over trials
    #self.fitMax  = [] # Best fitness over trials
    self.rank    = []
    self.birth   = []
    self.species = []
    
    self.order = None

    self.recurrence = recurrence
    firstInput = np.where(node[1,:]==1)[0][0]
    self.inputActivation = node[2,firstInput]
    self.nodeActivations = None
    self.wMultiplier = 1 # used for wann evaluation
    

  def clearState(self):
    self.nodeActivations = None

  def act(self, inPattern):
    if self.order is None: self.express()
    if self.order is None or False: return False

    nNodes = self.node.shape[1]
    nInput = self.nInput
    nOutput = self.nOutput
    wMat = self.wMat
    wMat[np.isnan(wMat)]=0

    # Vectorize input
    if np.ndim(inPattern) > 1:
        nSamples = np.shape(inPattern)[0]
    else:
        nSamples = 1



    lastActivations = self.nodeActivations
    if lastActivations is None: lastActivations = np.zeros(nNodes)

    nOutputOffset = 0
    recPlaceholders = []


    nodeAct  = np.zeros((nSamples,nNodes))
    nodeAct[:,0] = 1 # Bias activation

    if self.recurrence == 1:
      recurrenceNodes = np.where(self.node[1, self.order] == 5)[0]
      # nodeAct[:,nInput+1:len(recurrenceNodes)] = lastActivations[recurrenceNodes]
      inPattern = np.hstack((inPattern, lastActivations[recurrenceNodes]))

      nOutput += len(recurrenceNodes)
    elif self.recurrence == 2:
      # recPlaceholders = np.where(self.node[1, self.order] == 5)[0]
      recPlaceholders = np.where((self.node[1] == 5))[0]

      recurrenceNodes = np.array([])
      # print(self.order)
      # print(self.node)
      # print(self.conn)

      for idx in recPlaceholders:
        connI = np.where((self.conn[2] == self.node[0][idx]) * self.conn[4])[0]
        connI = connI.astype(int)
        connSrcs = self.conn[1][connI].astype(int)
        recoded = [ np.where(self.node[0] == s)[0][0] for s in connSrcs]
        # print(connSrcs)
        # print(recoded)
        recurrenceNodes = np.hstack((recurrenceNodes,
          lastActivations[self.order[recoded]]))

      inPattern = np.hstack((inPattern, recurrenceNodes))

      nOutputOffset = len(recPlaceholders)

    # Run input pattern through ANN    
    
    nodeAct[:,1:nInput+1] = inPattern




    # Propagate signal through hidden to output nodes
    iNode = nInput+1
    for iNode in range(nInput+1,nNodes):
        rawAct = np.dot(nodeAct, wMat[:,iNode]).squeeze()
        nodeAct[:,iNode] = applyAct(self.aVec[iNode], rawAct) 
    outStart = nNodes - (nOutput + nOutputOffset)
    outEnd = nNodes - nOutputOffset
    output = nodeAct[:,outStart:outEnd]   

    
    if np.all(wMat == 0):
      output -= 10

    self.nodeActivations = nodeAct[0]

    return output




  def nConns(self):
    """Returns number of active connections
    """
    return int(np.sum(self.conn[4,:]))

  def express(self):
    """Converts genes to weight matrix and activation vector
    """
    order, wMat = getNodeOrder(self.node, self.conn)
    self.order = order
    if order is not False:
      self.wMat = wMat*self.wMultiplier
      self.aVec = self.node[2,order]

      wVec = self.wMat.flatten()
      wVec[np.isnan(wVec)] = 0
      self.wVec  = wVec
      # self.nConn = np.sum(wVec!=0)
      return True
    else:
      return False

  def createChild(self, p, innov, gen=0, mate=None):
    """Create new individual with this individual as a parent

      Args:
        p      - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
        innov  - (np_array) - innovation record
           [5 X nUniqueGenes]
           [0,:] == Innovation Number
           [1,:] == Source
           [2,:] == Destination
           [3,:] == New Node?
           [4,:] == Generation evolved
        gen    - (int)      - (optional) generation (for innovation recording)
        mate   - (Ind)      - (optional) second for individual for crossover


    Returns:
        child  - (Ind)      - newly created individual
        innov  - (np_array) - updated innovation record

    """
    print("hellohellohello")
    if mate is not None:
      child = self.crossover(mate)
    else:
      child = RecurrentInd(self.conn, self.node, recurrence=self.recurrence)


    child, innov = child.mutate(p,innov,gen)


    return child, innov

# -- Canonical NEAT recombination operators ------------------------------ -- #

  def crossover(self,mate):
    """Combine genes of two individuals to produce new individual

      Procedure:
      ) Inherit all nodes and connections from most fit parent
      ) Identify matching connection genes in parentA and parentB
      ) Replace weights with parentB weights with some probability

      Args:
        parentA  - (Ind) - Fittest parent
          .conns - (np_array) - connection genes
                   [5 X nUniqueGenes]
                   [0,:] == Innovation Number (unique Id)
                   [1,:] == Source Node Id
                   [2,:] == Destination Node Id
                   [3,:] == Weight Value
                   [4,:] == Enabled?             
        parentB - (Ind) - Less fit parent

    Returns:
        child   - (Ind) - newly created individual

    """
    parentA = self
    parentB = mate

    # Inherit all nodes and connections from most fit parent
    child = RecurrentInd(parentA.conn, parentA.node, recurrence=self.recurrence)
    
    # Identify matching connection genes in ParentA and ParentB
    aConn = np.copy(parentA.conn[0,:])
    bConn = np.copy(parentB.conn[0,:])
    matching, IA, IB = np.intersect1d(aConn,bConn,return_indices=True)
    
    # Replace weights with parentB weights with some probability
    bProb = 0.5
    bGenes = np.random.rand(1,len(matching))<bProb
    child.conn[3,IA[bGenes[0]]] = parentB.conn[3,IB[bGenes[0]]]
    
    return child

  def mutate(self,p,innov=None,gen=None):
    """Randomly alter topology and weights of individual

    Args:
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
      child    - (Ind)      - individual to be mutated
        .conns - (np_array) - connection genes
                 [5 X nUniqueGenes] 
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?  
        .nodes - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved

    Returns:
        child   - (Ind)      - newly created individual
        innov   - (np_array) - innovation record

    """
    # Readability
    nConn = np.shape(self.conn)[1]
    connG = np.copy(self.conn)
    nodeG = np.copy(self.node)

    
    # - Re-enable connections
    disabled  = np.where(connG[4,:] == 0)[0]
    reenabled = np.random.rand(1,len(disabled)) < p['prob_enable']
    connG[4,disabled] = reenabled
         
    # - Weight mutation
    # [Canonical NEAT: 10% of weights are fully random...but seriously?]
    mutatedWeights = np.random.rand(1,nConn) < p['prob_mutConn'] # Choose weights to mutate
    weightChange = mutatedWeights * np.random.randn(1,nConn) * p['ann_mutSigma']
    connG[3,:] += weightChange[0]
    
    # Clamp weight strength [ Warning given for nan comparisons ]  
    connG[3, (connG[3,:] >  p['ann_absWCap'])] =  p['ann_absWCap']
    connG[3, (connG[3,:] < -p['ann_absWCap'])] = -p['ann_absWCap']
    
    if (np.random.rand() < p['prob_addNode']) and np.any(connG[4,:]==1):
      connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p)
    
    if (np.random.rand() < p['prob_addConn']):
      connG, nodeG, innov = self.mutAddConn(connG, nodeG, innov, gen, p) 
    
    child = RecurrentInd(connG, nodeG, recurrence=self.recurrence)
    child.birth = gen

    return child, innov

  def mutAddNode(self, connG, nodeG, innov, gen, p):
    """Add new node to genome

    Args:
      connG    - (np_array) - connection genes
                 [5 X nUniqueGenes] 
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?  
      nodeG    - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved
      gen      - (int)      - current generation
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)

    Returns:
      connG    - (np_array) - updated connection genes
      nodeG    - (np_array) - updated node genes
      innov    - (np_array) - updated innovation record

    """
    
    if innov is None:
      newNodeId = int(max(nodeG[0,:]+1))
      newConnId = connG[0,-1]+1    
    else:
      newNodeId = int(max(innov[2,:])+1) # next node id is a running counter
      newConnId = innov[0,-1]+1 
       
    # Choose connection to split
    connActive = np.where(connG[4,:] == 1)[0]
    if len(connActive) < 1:
      return connG, nodeG, innov # No active connections, nothing to split
    connSplit  = connActive[np.random.randint(len(connActive))]
    
    # Create new node
    newActivation = p['ann_actRange'][np.random.randint(len(p['ann_actRange']))]
    newNode = np.array([[newNodeId, 3, newActivation]]).T
    
    # Add connections to and from new node
    # -- Effort is taken to minimize disruption from node addition:
    # The 'weight to' the node is set to 1, the 'weight from' is set to the
    # original  weight. With a near linear activation function the change in performance should be minimal.

    connTo    = connG[:,connSplit].copy()
    connTo[0] = newConnId
    connTo[2] = newNodeId
    connTo[3] = 1 # weight set to 1
      
    connFrom    = connG[:,connSplit].copy()
    connFrom[0] = newConnId + 1
    connFrom[1] = newNodeId
    connFrom[3] = connG[3,connSplit] # weight set to previous weight value   
        
    newConns = np.vstack((connTo,connFrom)).T
        
    # Disable original connection
    connG[4,connSplit] = 0
        
    # Record innovations
    if innov is not None:
      newInnov = np.empty((5,2))
      newInnov[:,0] = np.hstack((connTo[0:3], newNodeId, gen))   
      newInnov[:,1] = np.hstack((connFrom[0:3], -1, gen)) 
      innov = np.hstack((innov,newInnov))
      
    # Add new structures to genome
    nodeG = np.hstack((nodeG,newNode))
    connG = np.hstack((connG,newConns))
    
    return connG, nodeG, innov

  def mutAddConn(self, connG, nodeG, innov, gen, p):
    """Add new connection to genome.
    To avoid creating recurrent connections all nodes are first sorted into
    layers, connections are then only created from nodes to nodes of the same or
    later layers.


    Todo: check for preexisting innovations to avoid duplicates in same gen

    Args:
      connG    - (np_array) - connection genes
                 [5 X nUniqueGenes] 
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?  
      nodeG    - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved
      gen      - (int)      - current generation
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
      connG    - (np_array) - updated connection genes
      innov    - (np_array) - updated innovation record

    """
    newConnId = connG[0,-1]+1
    # if innov is None:
    #   newConnId = connG[0,-1]+1
    # else:
    #   newConnId = innov[0,-1]+1


    """
    This code seems extremely obtuse to me but what it does is:
    - get topological order of nodes
    - assign layer to each node
    - create 2d array to capture (node id, layer)
    - create a permuted order of this array
    - in this order, for each node n
      - find all nodes with a layer > layer_n
      - remove nodes that already have a connection with n
      - if more than one possible destination remains, create connection and break loop.
    """

    nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
    nOuts = len(nodeG[0,nodeG[1,:] == 2]) + len(nodeG[0,nodeG[1,:] == 5])
    order, wMat = getNodeOrder(nodeG, connG)   # Topological Sort of Network
    hMat = wMat[nIns:-nOuts,nIns:-nOuts]
    hLay = getLayer(hMat)+1

    # To avoid recurrent connections nodes are sorted into layers, and connections are only allowed from lower to higher layers
    if len(hLay) > 0:
      lastLayer = max(hLay)+1
    else:
      lastLayer = 1
    L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer) ]
    nodeKey = np.c_[nodeG[0,order], L] # Assign Layers

    sources = np.random.permutation(len(nodeKey))
    for src in sources:
      srcLayer = nodeKey[src,1]
      dest = np.where(nodeKey[:,1] > srcLayer)[0]
      
      # Finding already existing connections:
      #   ) take all connection genes with this source (connG[1,:])
      #   ) take the destination of those genes (connG[2,:])
      #   ) convert to nodeKey index (Gotta be a better numpy way...)   
      srcIndx = np.where((connG[1,:]==nodeKey[src,0])*connG[4])[0]
      exist = connG[2,srcIndx]
      existKey = []
      for iExist in exist:
        existKey.append(np.where(nodeKey[:,0]==iExist)[0])
      dest = np.setdiff1d(dest,existKey) # Remove existing connections
      
      # Add a random valid connection
      np.random.shuffle(dest)
      if len(dest)>0:  # (there is a valid connection)
        connNew = np.empty((5,1))
        connNew[0] = newConnId
        connNew[1] = nodeKey[src,0]
        connNew[2] = nodeKey[dest[0],0]
        connNew[3] = (np.random.rand()-0.5)*2*p['ann_absWCap']
        connNew[4] = 1
        connG = np.c_[connG,connNew]

        destId = np.where(nodeG[0] == connNew[2][0])[0][0]
        # print("x%s %s"%(connNew[2][0], destId))
        # print(nodeG)
        # print(order)
        # print(nodeKey)
        # print(connG)
        if (self.recurrence == 2 and nodeG[1, destId] == 5):
        
          newNodeId = int(max(nodeG[0,:]+1))
          # if innov is None:
          # else:
          #   newNodeId = int(max(innov[2,:])+1) # next node id is a running counter

          newActivation = self.inputActivation # p['ann_actRange'][np.random.randint(len(p['ann_actRange']))]

          nInput = np.sum(nodeG[1] == 1) + 1
          newNode = np.array([[newNodeId, 1, newActivation]]).T
          nodeG = np.hstack((nodeG[:,:nInput], newNode, nodeG[:,nInput:]))
          # do we need to record the new node as an innovation?

          outs = nodeG[0, np.where((nodeG[1] == 2) + (nodeG[1] == 5))[0]]

          nRecCons = len(outs)
          recurrenceConns = np.empty((5, nRecCons,), dtype='d')
          recurrenceConns[0,:] = newConnId + 1 + np.arange(0,nRecCons,1)    # Connection Id
          recurrenceConns[1,:] = np.full(nRecCons, newNodeId) # Source Nodes
          recurrenceConns[2,:] = outs # Destination Nodes
          recurrenceConns[3,:] = np.nan                  # Weight Values
          recurrenceConns[4,:] = 0                       # not enabled
          connG = np.hstack((connG, recurrenceConns))

        # Record innovation
        if innov is not None:
          newInnov = np.hstack((connNew[0:3].flatten(), -1, gen))
          innov = np.hstack((innov,newInnov[:,None]))
        break

    return connG, nodeG, innov
