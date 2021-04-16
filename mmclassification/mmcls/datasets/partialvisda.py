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
class PartialVisDA(BaseDataset):

    CLASSES = [
        'aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person',
        'plant', 'skateboard', 'train', 'truck'
    ]

    def load_annotations(self):
        data_infos = []
        file_list = open(osp.join(self.data_prefix, 'image_list.txt'), 'r')
        imgs = file_list.readlines()
        mask = np.load(osp.join(self.data_prefix,'correct_idx.npy'))[0]
        for idx in mask:
            img = imgs[idx]
        #for img in imgs[mask]:
            img_prefix, label = img.replace('\n', '').split(' ')
            info = {'img_prefix': self.data_prefix} 
            info['img_info'] = {'filename': img_prefix}
            info['gt_label'] = np.array(int(label), dtype=np.int64)
            data_infos.append(info)
        return data_infos
