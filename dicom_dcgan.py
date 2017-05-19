import os
import dicom
import numpy as np
import random
#from keras.datasets import cifar10
from PIL import Image

outh = 64
outw = 64

class lung_data:
    def __init__(self):
        IMAGE_PATH = "sample_images\\"

        self.fpaths = []
        for root, dirs, files in os.walk(IMAGE_PATH, topdown=False):
            for name in files:
                fpath = os.path.join(root, name)
                self.fpaths.append(fpath)
                #break
                
        self.voxels = [dicom.read_file(img) for img in self.fpaths]
        self.voxels.sort(key=lambda x: int(x.InstanceNumber))

        #(self.voxels, _), (_, _) = cifar10.load_data()
        #m = [self.voxels[i].pixel_array for i in range(len(self.voxels))]
        m = [Image.fromarray(self.voxels[i].pixel_array) for i in range(len(self.voxels))]
        l = []
        #for i in range(len(self.m)):
        #    img = self.m[i]
        #    if img.mode.endswith("16"):
        #        continue
        #    l.append(img.thumbnail(size))
        #self.data = l
        for img in m:
            if not img.mode.endswith("16"):
                img.thumbnail((outh, outw))
                l.append(img)
        #self.m = [img.thumbnail((outh, outw)) for img in m if not img.mode.endswith("16")]
        m = l
        m = [np.array(img.convert("L")) for img in m]
        self.data = [np.reshape(m[i], (outh, outw, 1)) for i in range(len(m))]
        #self.data = m
        #random.seed(9001)
        #random.shuffle(self.data)
        self.itr = 0
        self.totalsz = len(self.data)
        
    def next_batch(self, bsize):
        if self.itr + bsize > self.totalsz:
            self.itr = 0
        nextb = self.data[self.itr:self.itr+bsize]
        self.itr = self.itr+bsize
        return nextb
        #return random.sample(self.data, bsize)
        
if __name__ == "__main__":
    data = lung_data()
    #size = 128, 128
    #data1 = data.next_batch(1)[0]
    #img = Image.fromarray(data1)
    # img.show()
    # img.thumbnail(size)
    # img.show()
