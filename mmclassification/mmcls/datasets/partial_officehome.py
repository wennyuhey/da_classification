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
class PartialOfficeHome(BaseDataset):

    CLASSES = [
        'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
        'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan',
        'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',
        'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
        'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
        'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone',
        'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam'
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
