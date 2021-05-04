import codecs
import os
import os.path as osp

import numpy as np
import torch

from .classwise import ClasswiseDADataset
from .builder import DATASETS
from .utils import download_and_extract_archive, rm_suffix
import scipy.io as sio


@DATASETS.register_module()
class ClasswiseVisDA(ClasswiseDADataset):

    CLASSES = [
        'aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person',
        'plant', 'skateboard', 'train', 'truck'
    ]


    def load_annotations(self, domain):
        data_infos = []
        file_list = open(osp.join(self.data_prefix, domain, 'image_list.txt'), 'r')
        imgs = file_list.readlines()
        class_list = np.zeros(len(self.CLASSES))
        for img in imgs:
            img_prefix, label = img.replace('\n', '').split(' ')
            info = {'img_prefix': osp.join(self.data_prefix, domain)}
            info['img_info'] = {'filename': img_prefix}
            info['gt_label'] = np.array(int(label), dtype=np.int64)
            info['pseudo_label'] = np.array(-1, dtype=np.int64)
            class_list[int(label)] += 1
            data_infos.append(info)
        import pdb
        pdb.set_trace()
        return data_infos, class_list
