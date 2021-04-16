import copy
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmcls.models.losses import accuracy
from .pipelines import Compose
import random


class CategoricalDADataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self,
                 data_prefix,
                 pipeline,
                 samples_per_class,
                 times=2,
                 class_set=None,
                 classes=None,
                 ann_file_s=None,
                 ann_file_t=None,
                 test_mode=False):
        super(CategoricalDADataset, self).__init__()

        self.ann_file_s = ann_file_s
        self.ann_file_t = ann_file_t
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = {'source': self.load_annotations('source'), 'target': self.load_annotations('target')}
        self.CLASSES = self.get_classes(classes)
        self.batch_size = samples_per_class
        self.class_set = class_set
        self.data = []
        self.times = times

    @abstractmethod
    def load_annotations(self, domain):
        pass

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return self.data_infos[idx]['gt_label'].astype(np.int)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.CLASSES)

    def __getitem__(self, idx):
        data = {}
        for d in ['source', 'target']:
            data_info = self.data_infos[d]
            sample_idx = random.sample(range(len(data_info[idx])), self.batch_size)
            tmp = {'img': [], 'gt_label': []}
            for i in range(self.times):
                img = []
                label = []
                for i in sample_idx:
                    data_trans = self.pipeline(copy.deepcopy(data_info[idx][i]))
                    img.append(data_trans['img'])
                    label.append(data_trans['gt_label'].reshape(1))
                tmp['img'].append(img)
            tmp['gt_label'] = label
            data[d] = tmp                
        return data

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def update(self,
        target_label=None,
        class_set=None):
        if class_set is None:
            self.class_set = self.CLASSES
        else:
            self.class_set = class_set

        if target_label is not None:
            pass

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options={'topk': (1, 5)},
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['accuracy']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        if metric == 'accuracy':
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(results, gt_labels, topk)
            if isinstance(topk, tuple):
                eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            elif isinstance(topk, int):
                eval_results = {f'top-{topk}': acc.item()}
        return eval_results
