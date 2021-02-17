# DQN-Vanilla For Atari Games

Implementation of DQN on pytorch and other tools. 

## First Version
Accepts all ALE environments though gym. Can use ray to initialize multiple actors to contribute to the memory replay. Some defaults are based on the ones published on the 2013's Nature Letter.
This by default uses adam from pytorch as rmsprop could be implemented in a different way that does not brings the same results as the authors of such Letter. 

The code is ugly now.
