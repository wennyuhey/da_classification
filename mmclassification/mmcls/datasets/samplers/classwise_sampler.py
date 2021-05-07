from torch.utils.data import Sampler
import random
import numpy as np


class ClasswiseSampler(Sampler):

    def __init__(self, dataset, class_num, batch_size):
        self.bs_per_class = batch_size
        self.class_sample = class_num
        self.batch_size = self.bs_per_class * self.class_sample
        #self.num_class= dataset.source_cls.shape[0] # number of class in the dataset
        self.num_class = dataset.num_classes
        self.samples_per_class = dataset.class_sample # number of image per class
        self.batch_num = int(self.samples_per_class//self.bs_per_class) #batch number per class
        self.class_len = self.batch_num * self.bs_per_class # numbers of samples used per class
        self.length = int(self.class_len * self.num_class / self.batch_size)
        self.real_len = self.length * self.batch_size


    def __iter__(self):
        self.idx = np.arange(self.num_class * self.samples_per_class)
        self.idx = list(self.idx.reshape(self.num_class, self.samples_per_class))
        random.shuffle(self.idx)
        for i in self.idx:
            random.shuffle(i)
        self.idx = np.array(self.idx)
        self.idx = self.idx[:, 0: self.class_len]
        self.idx = self.idx.reshape(-1, self.batch_num, self.bs_per_class)
        self.idx = self.idx.transpose(1, 0, 2).flatten()
        self.idx = np.array_split(self.idx[0:self.real_len], int(self.length))
        self.idx = [i.tolist() for i in self.idx]
        return iter(self.idx)

    def __len__(self):
        return int(self.class_len * self.num_class / self.batch_size)

from torch.utils.data import DistributedSampler as _DistributedSampler

class DistributedClasswiseSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 class_num,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 seed=None,
                 shuffle=True):

        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.bs_per_class = batch_size
        self.num_class = dataset.num_classes
        self.samples_per_class = dataset.class_sample
        self.total_iter = int(self.samples_per_class / (self.num_replicas * self.bs_per_class))
        self.class_len = self.total_iter * self.num_replicas * self.bs_per_class
        self.replicas_len = self.total_iter * self.bs_per_class * self.num_class
    def __iter__(self):
        idx = np.arange(self.num_class * self.samples_per_class).reshape(self.num_class, -1)
        idx = list(idx)
        random.shuffle(idx)
        for i in idx:
            random.shuffle(i)
        idx = np.array(idx)
        idx = idx[:, 0: self.class_len]
        idx = idx.reshape(self.num_class, self.num_replicas, -1)
        idx = idx.transpose(1, 0, 2)
        idx = idx.reshape(self.num_replicas, self.num_class, -1, self.bs_per_class)
        idx = idx.transpose(0, 2, 1, 3).reshape(self.num_replicas, -1)
        #indices = idx[self.rank * self.replicas_len : (self.rank + 1) * self.replicas_len]
        indices = idx[self.rank]
        return iter(indices)


    def __len__(self):
        return self.replicas_len
