import torch
import numpy as np
import numpy.random as rdm
from CONST import *

class MemoryReplay(object):
    """
    Main Storage for the transitions experienced by the actors.

    It has methods to Sample, combineBuffers

    Parameters
    ----------
    capacity: int
        Number of transitions to store
    """
    def __init__(self,
                 capacity:int = MEMORY_SIZE,
                 state_shape:list = [84, 84],
                 LHist:int = LHIST,
                 state_dtype_in:np.dtype = np.uint8,
                 state_dtype_out:np.dtype = np.float32,
                 action_dtype_in:np.dtype = np.uint8,
                 action_dtype_out:torch.dtype = torch.int64,
                 reward_dtype_in:np.dtype = np.float32,
                 reward_dtype_out:torch.dtype = torch.float32,
                 ):
        
        self.s_in_shape = state_shape
        self.s_dtype_in = state_dtype_in
        self.s_dtype = state_dtype_out
        self.a_dtype = action_dtype_out
        self.r_dtype = reward_dtype_out

        self.capacity = capacity
        self.LHist = LHist
        self.shapeHistOut = [LHist] + state_shape
        self._i = 0
        self.FO = False

        self.s_buffer = np.zeros([capacity]+state_shape, dtype = state_dtype_in)
        self.a_buffer = np.zeros(capacity, dtype = action_dtype_in)
        self.r_buffer = np.zeros(capacity, dtype = reward_dtype_in)
        self.t_buffer = np.ones(capacity, dtype = np.bool_) # Inverse logic

    def add(self, s, a, r, t):
        """
        Add one item
        """
        self.s_buffer[self._i] = s
        self.a_buffer[self._i] = a
        self.r_buffer[self._i] = r
        self.t_buffer[self._i] = t
        self._i = (self._i + 1) % self.capacity
        if self._i == 0:
            self.FO = True

    def get2History(self, i:int, m:int, st1, st2):
        # modify inplace
        for n, j in enumerate(range(i, i - self.LHist - 1, -1)):
            s, _, _, t = self[j]
            if n < self.LHist:
                st2[m][n] = s
            if n > 0:
                st1[m][n - 1] = s
            if not t and n >= 0:
                # This should happend rarely
                break

    def __getitem__(self, i:int):
        if i < self._i or self.FO:
            i = i % self.capacity
            return (self.s_buffer[i],
                    self.a_buffer[i],
                    self.r_buffer[i],
                    self.t_buffer[i])
        else:
            return self.zeroe

    @property
    def zeroe(self):
        return (np.zeros(self.s_in_shape, dtype=self.s_dtype_in),
                0,
                0.0,
                False)

    def Sample(self, mini_batch_size:int, tDevice = torch.device('cpu')):
        """
        Process and returns a mini batch. The tuple returned are
        all torch tensors.
        
        If tDevice is cpu class, this process may consume more cpu resources
        than expected. Could be detrimental if hosting multiple instances. 
        This seems expected from using torch. (Y)

        Parameters
        ---------
        mini_batch_size: int
            Number of samples that compose the mini batch
        tDevice: torch.device
            Optional. Torch device target for the mini batch
            to reside on.
        """
        assert mini_batch_size > 0, "The size of the mini batch must be positive"

        if self._i > mini_batch_size + self.LHist or self.FO:
            #ids = rdm.choice(np.arange(self.LHist, self.capacity if self.FO else self._i - 1), 
            #                        size=mini_batch_size, replace=False)
            ids = rdm.randint(self.LHist, self.capacity if self.FO else self._i - 1, 
                                    size=mini_batch_size)
            st1 = np.zeros([mini_batch_size] + self.shapeHistOut, 
                           dtype = self.s_dtype)
            st2 = st1.copy()
            for m, i in enumerate(ids):
                for n, j in enumerate(range(i, i - self.LHist - 1, -1)):
                    s, _, _, t = self[j]
                    if n < self.LHist:
                        st2[m][n] = s.copy()
                    if n > 0:
                        st1[m][n - 1] = s.copy()
                    if not t and n >= 0:
                        # This should happend rarely
                        break
            at = self.a_buffer[ids]
            rt = self.r_buffer[ids]
            terminals = self.t_buffer[ids].astype(np.float32)
            # Passing to torch format
            st1 = torch.as_tensor(st1, device=tDevice).div(255).requires_grad_()
            st2 = torch.as_tensor(st2, device=tDevice).div(255)
            terminals = torch.as_tensor(terminals, dtype=torch.float32, device=tDevice)
            at = torch.as_tensor(at, dtype=self.a_dtype, device=tDevice)
            rt = torch.as_tensor(rt, dtype=self.r_dtype, device=tDevice)
            return (st1, at, rt, st2, terminals)
        else:
            raise IndexError("The memory does not contains enough transitions to generate the sample")

    def combineBuffers(self, *buffers):
        """
        Pass any number of tuples with buffers to add to the
        memory replay.
        """
        for buffer in buffers:
            # Unpack
            st_other, at_other, rt_other, t_other = buffer
            # Adding a zero between buffers
            if len(buffers) > 1:
                self.add(*self.zeroe)
            # Update buffers
            l = t_other.shape[0]
            if l + self._i > self.capacity:
                ovr = (l + self._i) % self.capacity
                ovrl = (self.capacity - self._i)
                self.s_buffer[self._i:] = st_other[:ovrl]
                self.a_buffer[self._i:] = at_other[:ovrl]
                self.r_buffer[self._i:] = rt_other[:ovrl]
                self.t_buffer[self._i:] = t_other[:ovrl]
                self.s_buffer[:ovr] = st_other[ovrl:]
                self.a_buffer[:ovr] = at_other[ovrl:]
                self.r_buffer[:ovr] = rt_other[ovrl:]
                self.t_buffer[:ovr] = t_other[ovrl:]
            else:
                self.s_buffer[self._i:self._i + l] = st_other
                self.a_buffer[self._i:self._i + l] = at_other
                self.r_buffer[self._i:self._i + l] = rt_other
                self.t_buffer[self._i:self._i + l] = t_other
            ax = (self._i + l)
            if ax >= self.capacity:
                self.FO = True
            self._i = ax % self.capacity
        del buffers

    def __len__(self):
        if self.FO:
            return self.capacity
        else:
            return self._i

    def dictToSave(self):
        this = dict()
        if self.capacity > 6*10**5:
            # Dic to big
            return None
        this['s_buffer'] = self.s_buffer
        this['a_buffer'] = self.a_buffer
        this['r_buffer'] = self.r_buffer
        this['t_buffer'] = self.t_buffer
        this['i'] = self._i
        this['FilledOnce'] = self.FO
        this['capacity'] = self.capacity
        return this

    def loadFromDict(self, this):
        try:
            self.s_buffer = this['s_buffer']
            self.a_buffer = this['a_buffer']
            self.r_buffer = this['r_buffer']
            self.t_buffer = this['t_buffer']
            self._i = this['i']
            self.FO = this['FilledOnce']
            self.capacity = this['capacity']
            print("Successfully loading Buffer from dict")
        except:
            print("Error loading Buffer loaded from dict")

    def showBuffer(self, samples:int = 20, Wait:int = 3):
        import matplotlib.pyplot as plt
        # Drawing samples
        Samplei = np.random.randint(self._i if not self.FO else self.capacity, size=samples)
        for i in Samplei:
            plt.ion()
            fig = plt.figure(figsize=(10,3))
            plt.title('Non-terminal' if self.t_buffer[i] else 'Terminal')
            plt.axis('off')
            for n, j in enumerate(range(i, i - self.LHist, -1)):
                fig.add_subplot(1, 4, n + 1)
                plt.imshow(self.s_buffer[j])
                plt.axis('off')
            plt.pause(Wait)
            plt.close(fig)