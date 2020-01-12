import numpy as np
import copy
from .ind_recurrent import RecurrentInd
from .ann import getLayer, getNodeOrder
from utils import listXor


class RecurrentWannInd(RecurrentInd):
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
             [1,:] == Type (1=input, 2=output, 3=hidden, 4=bias, 5=recurrence)
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
      fitMax  - (double)   - best fitness over all trials (higher better)
      rank    - (int)      - rank in population (lower better)
      birth   - (int)      - generation born
      species - (int)      - ID of species
    """
    RecurrentInd.__init__(self,conn,node, recurrence=recurrence)
    self.fitMax  = [] # Best fitness over trials

  def createChild(self, p, innov, gen=0):
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


    Returns:
        child  - (Ind)      - newly created individual
        innov  - (np_array) - updated innovation record

    """     
    child = RecurrentWannInd(self.conn, self.node, recurrence=self.recurrence)
        
    child.express()

    # print("Before")
    # print(child.node)
    # print(child.conn)

    child, innov = child.topoMutate(p,innov,gen)

    # child.express()
    # print("after")
    # print(child.node)
    # print(child.conn)
    return child, innov

# -- 'Single Weight Network' topological mutation ------------------------ -- #

  def topoMutate(self, p, innov,gen):
    """Randomly alter topology of individual
    Note: This operator forces precisely ONE topological change 

    Args:
      child    - (Ind) - individual to be mutated
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
    # nConn = np.shape(self.conn)[1]
    connG = np.copy(self.conn)
    nodeG = np.copy(self.node)

    # Choose topological mutation
    topoRoulette = np.array((p['prob_addConn'], p['prob_addNode'], \
                             p['prob_enable'] , p['prob_mutAct']))

    spin = np.random.rand()*np.sum(topoRoulette)
    slot = topoRoulette[0]
    choice = topoRoulette.size
    for i in range(1,topoRoulette.size):
      if spin < slot:
        choice = i
        break
      else:
        slot += topoRoulette[i]

    # Add Connection
    if choice is 1:
      connG, nodeG, innov = self.mutAddConn(connG, nodeG, innov, gen, p)  

    # Add Node
    elif choice is 2:
      connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p)

    # Enable Connection
    elif choice is 3:
      pass
      # disabled = np.where(connG[4,:] == 0)[0]
      # if len(disabled) > 0:
      #   enable = np.random.randint(len(disabled))
      #   connG[4,disabled[enable]] = 1

    # Mutate Activation
    elif choice is 4:
      start = 1+self.nInput + self.nOutput
      end = nodeG.shape[1]           
      if start != end:
        mutNode = np.random.randint(start,end)
        newActPool = listXor([int(nodeG[2,mutNode])], p['ann_actRange'])
        nodeG[2,mutNode] = int(newActPool[np.random.randint(len(newActPool))])

    child = RecurrentWannInd(connG, nodeG, recurrence=self.recurrence)
    child.birth = gen

    return child, innov
