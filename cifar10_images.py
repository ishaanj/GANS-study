import os
import dicom
import numpy as np
import random
from keras.datasets import cifar10

class cifar_data:
    def __init__(self):
        (voxels, _),_ = cifar10.load_data()
        m = [voxels[i] for i in range(len(voxels))]
        self.data = [np.reshape(m[i], (3072,)) for i in range(len(m))]
        random.seed(9001)
        random.shuffle(self.data)
        self.itr = 0
        self.totalsz = len(self.data)
        
    def next_batch(self, bsize):
        #nextb = self.data[self.itr:self.itr+bsize]
        #self.itr = self.itr+bsize
        #return nextb
        return random.sample(self.data, bsize)
