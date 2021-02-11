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
class Office31(BaseDataset):

    CLASSES = [
        'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair',
        'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer',
        'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone',
        'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
        'tape_dispenser', 'trash_can'
    ]

    def load_annotations(self):
        data_infos = []
        domain_list = os.listdir(self.data_prefix)
        for domain in domain_list:
            domain_dir = osp.join(self.data_prefix, domain, 'images')
            label_list = os.listdir(domain_dir)
            for idx, label in enumerate(label_list):
                file_list = os.listdir(osp.join(domain_dir, label))
                for img in file_list:
                    info = {'img_prefix': domain_dir}
                    info['img_info'] = {'filename': osp.join(label, img)}
                    info['gt_label'] = np.array(idx, dtype=np.int64)
                    data_infos.append(info)
        """
        for idx, label in enumerate(label_list):
            file_list = os.listdir(osp.join(self.data_prefix, label, 'images'))
            for img in file_list:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': osp.join(label, img)}
                info['gt_label'] = np.array(idx, dtype=np.int64)
                data_infos.append(info)
        """
        return data_infos
