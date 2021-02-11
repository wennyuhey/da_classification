import codecs
import os
import os.path as osp

import numpy as np
import torch

from .base_dataset import BaseDataset
from .builder import DATASETS
from .utils import download_and_extract_archive, rm_suffix
import scipy.io as sio


@DATASETS.register_module()
class SVHN(BaseDataset):

    CLASSES = [
        '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five',
        '6 - six', '7 - seven', '8 - eight', '9 - nine'
    ]

    def load_annotations(self):
        train_file = osp.join(
            self.data_prefix, 'train_32x32.mat')
        test_file = osp.join(
            self.data_prefix, 'test_32x32.mat')

        train_set = sio.loadmat(train_file)
        test_set = sio.loadmat(test_file)

        if not self.test_mode:
            imgs = train_set['X'].transpose(3,0,1,2)
            gt_labels = train_set['y']
        else:
            imgs = test_set['X'].transpose(3,0,1,2)
            gt_labels = test_set['y']

        data_infos = []
        for img, gt_label in zip(imgs, gt_labels):
            if gt_label[0] == 10:
                continue
            img = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
            gt_label = np.array(gt_label[0], dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos
