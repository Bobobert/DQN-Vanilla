import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class dqnPolicy: # TODO
    def __init__(self, lHist:int = 4, actions:int = 18, dropouts:list=[0.5,0.5,0.5,0.5]):
        op_net = DQN(lHist, actions, dropouts)
        self.nets = OrderedDict()
        self.nets['op'] = op_net
        self.Id = 0
        self.lhist = lHist
        self.actions = actions
        self.drops = dropouts

    def updateNet(self, *grads):
        pass
    
    def avgGrads(self, *grads):
        if len(grads) == 0:
            print("No grads were passed")
            return None
        if len(grads) == 1:
            return grads[0]
        alpha = 1 / len(grads)
        avg_grads = []
        for T in grads[0]:
            avg_grads += [T.new_zeros(T.shape)]
        for grad in grads:
            for a, b in zip(avg_grads, grad):
                a.add(b, alpha=alpha)
        
    def clone(self, name:str=''):
        assert name != 'op', "Cannot use that name"
        newClone = self.nets['op'].copy()
        if name != '':
            newClone.name = name
        self.nets[newClone.name + str(self.Id)] = newClone # Save ref
        return newClone

class atariDQN(nn.Module):
    """
    Policy network for DQN-Atari

    parameters
    ----------
    lHist: int
        Number of frames on the stack for a history. The frame size 
        is defined as (84, 84)
    actions: int
        Number of actions in which the policy will chose
    dropouts: list
        A list with 4 probabilities each to decide for the layers from
        cv1, cv2, cv3 and fc1 if drops some nodes or not.

    """
    def __init__(self, lHist:int = 4, actions:int = 18, dropouts:list=[0.5,0.5,0.5,0.5]):
        super(atariDQN, self).__init__()
        assert len(dropouts) == 4, "No enough dropouts!"
        # Variables
        self.lHist = lHist
        self.outputs = actions
        self.drops = dropouts.copy()
        self.name = 'DQN-policy'
        # Operational
        self.dropOvr = False
        # Net
        self.rectifier = F.relu
        self.drop = F.dropout
        self.dropcv = F.dropout
        self.cv1 = nn.Conv2d(lHist, 32, 8, 4)
        self.cv2 = nn.Conv2d(32, 64, 4, 2)
        self.cv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, actions) # from fully connected to actions

    def forward(self, X):
        drop = True if self.dropOvr else self.training
        X = self.cv1(X)
        X = self.dropcv(self.rectifier(X), p=self.drops[0], training=drop)
        X = self.cv2(X)
        X = self.dropcv(self.rectifier(X), p=self.drops[1], training=drop)
        X = self.cv3(X)
        X = self.dropcv(self.rectifier(X), p=self.drops[2], training=drop)
        X = self.fc1(X.flatten(1))
        X = self.drop(self.rectifier(X), p=self.drops[3], training=drop)
        return self.fc2(X)
        
    def copy(self):
        new = atariDQN(self.lHist, self.outputs, self.drops)
        new.load_state_dict(self.state_dict())
        new.name = self.name + '.copy'
        return new

    def Dropout(self, activation = None):
        """
        Sets or changes the actual state of the dropout override. 
        If set to false, the modules follows the .train() and .eval()
        as expected. If true the dropout is always applied.
        """
        self.dropOvr = not self.dropOvr if activation is None else activation
        return None

    def initNet(self):
        # By default in torch 1.6 conv and linear layers are 
        # initilized with the kaiming_uniform_ or He
        pass

    def updateState(self, stateDict):
        self.load_state_dict(stateDict, strict=True)

    def getState(self, cpu:bool=False):
        if not cpu:
            return self.state_dict()
        else:
            stateDict = self.state_dict()
            for key in stateDict.keys():
                stateDict[key] = stateDict[key].to(torch.device('cpu'))
            return stateDict