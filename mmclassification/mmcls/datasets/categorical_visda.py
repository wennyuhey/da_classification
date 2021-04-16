import codecs
import os
import os.path as osp

import numpy as np
import torch

from .categorical import CategoricalDADataset
from .builder import DATASETS
from .utils import download_and_extract_archive, rm_suffix
import scipy.io as sio


@DATASETS.register_module()
class CategoricalVisDA(CategoricalDADataset):

    CLASSES = [
        'aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person',
        'plant', 'skateboard', 'train', 'truck'
    ]

    domain_folder = {'source': 'train', 'target': 'validation'}

    def load_annotations(self, domain):
        data_infos = {}
        domain_folder = {'source': 'train', 'target': 'validation'}
        for idx, _ in enumerate(self.CLASSES):
            data_infos[idx] = []
        file_list = open(osp.join(self.data_prefix, domain_folder[domain], 'image_list.txt'), 'r')
        imgs = file_list.readlines()
        for img in imgs:
            img_prefix, label = img.replace('\n', '').split(' ')
            info = {'img_prefix': osp.join(self.data_prefix, domain_folder[domain])}
            info['img_info'] = {'filename': img_prefix}
            info['gt_label'] = np.array(int(label), dtype=np.int64)
            data_infos[int(label)].append(info)
        return data_infos
