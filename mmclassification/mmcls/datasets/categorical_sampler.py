from torch.utils.data import Sampler
import random
import numpy as np


class CategoricalSampler(Sampler):

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
        self.idx = list(self.idx.reshape(self.num_class, self.samples_per_class)[:, 0: self.class_len])
        random.shuffle(self.idx)
        for i in self.idx:
            random.shuffle(i)
        self.idx = np.array(self.idx)
        self.idx = self.idx.reshape(-1, self.batch_num, self.bs_per_class)
        self.idx = self.idx.transpose(1, 0, 2).flatten()
        self.idx = np.array_split(self.idx[0:self.real_len], int(self.length))
        self.idx = [i.tolist() for i in self.idx]
        return iter(self.idx)

    def __len__(self):
        return int(self.class_len * self.num_class / self.batch_size)

