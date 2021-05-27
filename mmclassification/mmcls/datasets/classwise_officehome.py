import codecs
import mmcv
import os
import os.path as osp

import numpy as np
import torch

from .classwise import ClasswiseDADataset
from .builder import DATASETS
from .utils import download_and_extract_archive, rm_suffix
import scipy.io as sio


@DATASETS.register_module()
class ClasswiseOfficeHome(ClasswiseDADataset):

    CLASSES = [
        'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
        'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan',
        'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',
        'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
        'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
        'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone',
        'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam'
    ]

    def load_annotations(self, domain):
        data_infos = []
        domain_prefix = osp.join(self.data_prefix, domain)
        label_list = os.listdir(domain_prefix)
        class_list = np.zeros(len(self.CLASSES))
        for idx, label in enumerate(self.CLASSES):
            file_dir = osp.join(domain_prefix, label)
            file_list = sorted(os.listdir(file_dir))
            for img in file_list:
                info = {'img_prefix': domain_prefix}
                info['img_info'] = {'filename': osp.join(label, img)}
                info['gt_label'] = np.array(idx, dtype=np.int64)
                info['pseudo_label'] = np.array(-1, dtype=np.int64)
                data_infos.append(info)
                class_list[idx] += 1
        return data_infos, class_list
