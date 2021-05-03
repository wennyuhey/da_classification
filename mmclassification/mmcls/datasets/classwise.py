import copy
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmcls.models.losses import accuracy
from .pipelines import Compose
import random


class ClasswiseDADataset(Dataset, metaclass=ABCMeta):
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
                 times=2,
                 class_set=None,
                 classes=None,
                 load_mode='balanced',
                 source_prefix=None,
                 target_prefix=None,
                 test_mode=False):
        super(ClasswiseDADataset, self).__init__()
        self.data_prefix = data_prefix
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.num_classes = len(self.CLASSES)
        self.source_data, self.source_class_list = self.load_annotations(source_prefix)
        self.target_data, self.target_class_list = self.load_annotations(target_prefix)
        self.source_count = np.insert(np.cumsum(self.source_class_list), 0, 0).astype(int)
        self.target_count = np.insert(np.cumsum(self.target_class_list), 0, 0).astype(int)
        self.times = times
        self.load_mode = load_mode

        self.source = [] #[np.array, np.array, ..]classwise
        self.target = []

        if class_set is None:
            self.class_set = np.arange(len(self.CLASSES))
        else:
            self.class_set = class_set
        self.category_preprocess(**self.load_mode)

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

    def category_preprocess(self, target_balance, target_shuffle, source_balance, source_shuffle):
        assert not(target_balance==False and target_shuffle==False)
        assert not(source_balance==False and source_shuffle==False)
        del self.source
        del self.target
        self.source = []
        self.target = []
        self.class_sample = int(max(max(self.source_class_list[self.class_set]), max(self.target_class_list[self.class_set])))
        data_len = self.class_sample * len(self.class_set)
        if target_balance == True or source_balance == True:
            for cls in self.class_set:
                if source_balance: 
                    ori_s = np.arange(self.source_count[cls], self.source_count[cls+1])
                    ori_s_int = np.tile(ori_s, int(self.class_sample//self.source_class_list[cls]))
                    ori_s_res = np.array(random.choices(ori_s, k = int(self.class_sample%self.source_class_list[cls])))
                    self.source.append(np.hstack((ori_s_int, ori_s_res)))
                if target_balance:
                    ori_t = np.arange(self.target_count[cls], self.target_count[cls+1])
                    ori_t_int = np.tile(ori_t, int(self.class_sample//self.target_class_list[cls]))
                    ori_t_res = np.array(random.choices(ori_t, k = int(self.class_sample%self.target_class_list[cls])))
                    self.target.append(np.hstack((ori_t_int, ori_t_res)))

        if source_balance:
            self.source_cls = np.array(self.source).astype(int)
            self.source = self.source_cls.flatten()
        else:
            ori_s = np.arange(self.source_count[-1])
            ori_s_int = np.tile(ori_s, int(data_len//len(ori_s)))
            ori_s_res = np.array(random.choices(ori_s, k=int(data_len%len(ori_s))))
            self.source = np.hstack((ori_s_int, ori_s_res)).astype(int)

        if target_balance:
            self.target_cls = np.array(self.target).astype(int)
            self.target = self.target_cls.flatten()
        else:
            ori_t = np.arange(self.target_count[-1])
            ori_t_int = np.tile(ori_t, int(data_len//len(ori_t)))
            ori_t_res = np.array(random.choices(ori_t, k=int(data_len%len(ori_t))))
            self.target = np.hstack((ori_t_int, ori_t_res)).astype(int)

        if target_shuffle:
            random.shuffle(self.target)
        if source_shuffle:
            random.shuffle(self.source)

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
        return len(self.source)

    def __getitem__(self, idx):
        img_s = []
        img_t = []
        for i in range(self.times):
            img_s.append(self.pipeline(copy.deepcopy(self.source_data[self.source[idx]]))['img'])
            img_t.append(self.pipeline(copy.deepcopy(self.target_data[self.target[idx]]))['img'])
        data_s = self.pipeline(copy.deepcopy(self.source_data[self.source[idx]]))
        data_s['img'] = img_s
        data_t = self.pipeline(copy.deepcopy(self.target_data[self.target[idx]]))
        data_t['img'] = img_t
        data = {'source': data_s, 'target': data_t}
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
            for idx, label in enumerate(target_label):
                self.target_data[idx]['pseudo_label'] = np.array(label.item(), dtype=np.int64)

        #self.category_preprocess()

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
