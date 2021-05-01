import codecs
import mmcv
import os
import os.path as osp

import numpy as np
import torch

from .base_dataset import BaseDataset
from .builder import DATASETS
from .utils import download_and_extract_archive, rm_suffix
import scipy.io as sio


@DATASETS.register_module()
class PartialOffice(BaseDataset):

    CLASSES = [
        'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair',
        'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer',
        'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone',
        'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
        'tape_dispenser', 'trash_can'
    ]

    def load_annotations(self):
        data_infos = []
        for idx, label in enumerate(self.CLASSES):
            file_dir = osp.join(self.data_prefix, label)
            file_list = sorted(os.listdir(file_dir))
            for img in file_list:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': osp.join(label, img)}
                info['gt_label'] = np.array(idx, dtype=np.int64)
                data_infos.append(info)
        return data_infos
