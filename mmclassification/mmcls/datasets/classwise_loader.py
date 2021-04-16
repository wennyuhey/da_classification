from torch.utils.data import RandomSampler, BatchSampler, DataLoader
import torch
from math import ceil as ceil
import numpy as np

def collate_fn(data):
    # data is a list: index indicates classes
    num_classes = len(data)
    times = len(data[0]['source']['img'])

    """
    tmp_data = {'source': {'img': [], 'gt_label': []}, 'target': {'img': [], 'gt_label': []}}
    for i in range(num_classes):
        for d in ['source', 'target']:
            imgs = data[i][d]['img']
            labels = data[i][d]['gt_label']
            for j in range(times):
                tmp = []
                tmp_label = []
                for img in imgs:
                    tmp.append(img[j])
                    
                tmp_data[d]['img'].append(tmp)
 
            for label in labels:
                tmp_data[d]['gt_label'].append(label.reshape(1))

    tmp_data['source']['gt_label'] = torch.cat((tmp_data['source']['gt_label']))
    tmp_data['target']['gt_label'] = torch.cat((tmp_data['target']['gt_label']))
    """
    data_collate = {'source':{'img':[], 'gt_label':[]}, 'target':{'img':[], 'gt_label':[]}}
    for i in range(times):
        tmp = {'source':{'img':[], 'gt_label':[]}, 'target':{'img':[], 'gt_label':[]}}
        for label in range(num_classes):
           for d in ['source', 'target']:
               imgs = data[label][d]['img']
               labels = data[label][d]['gt_label']
               tmp[d]['img']+=[j.numpy() for j in imgs[i]]
               tmp[d]['gt_label']+=labels
        test = torch.as_tensor(tmp['source']['img'])
        data_collate['source']['img'].append(torch.from_numpy(np.array(tmp['source']['img'])))
        data_collate['target']['img'].append(torch.from_numpy(np.array(tmp['target']['img'])))
    data_collate['source']['gt_label'] = torch.cat(tuple(tmp['source']['gt_label']))
    data_collate['target']['gt_label'] = torch.cat(tuple(tmp['target']['gt_label']))
    return data_collate

class CategoricalDataLoader(object):
    def name(self):
        return 'CategoricalDataLoader'

    def __init__(self, dataset,
                class_set=[], num_selected_classes=0,
                seed=None, num_workers=0, drop_last=True,
                **kwargs):

        # dataset type
        self.dataset = dataset
        from mmcls.datasets import SupConDataset
        if isinstance(self.dataset, SupConDataset):
            self.original_dataset = self.dataset.dataset
        else:
            self.original_dataset = self.dataset
 

        # dataset parameters
        self.class_set = class_set
        self.seed = seed

        # loader parameters
        self.num_selected_classes = num_selected_classes
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.kwargs = kwargs

    def construct(self, target_label, class_set):
        self.class_set = class_set

        self.original_dataset.update(target_label=target_label,
                  class_set=self.class_set,
                  **self.kwargs)

        drop_last = self.drop_last
        sampler = RandomSampler(self.dataset)
        batch_sampler = BatchSampler(sampler,
                                 self.num_selected_classes, drop_last)

        self.dataloader = DataLoader(self.dataset,
                         batch_sampler=batch_sampler,
                         collate_fn=collate_fn,
                         num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        for c in range(len(self.original_dataset.class_set)):
            c_len = max([len(self.original_dataset.data_infos[d][c]) // \
                  self.original_dataset.batch_size for d in ['source', 'target']])
            dataset_len += c_len

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len

