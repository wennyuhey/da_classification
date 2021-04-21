import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader
import numpy as np

class DatasetHook(Hook):

    def __init__(self, dataloader, interval=1, by_epoch=True, **kwargs):
        #if not isinstance(dataloader, DataLoader):
        #    raise TypeError('dataloader must be a pytorch DataLoader, but got'
        #                    f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch

    def before_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import da_single_gpu_test
        results = da_single_gpu_test(runner.model, self.dataloader, bar_show=False, show=False)
        results = np.vstack(results)
        #pred_label = results.argsort(axis=1)[:, -1:][:, ::-1].squeeze().flatten()
        runner.data_loader.dataset.update(results)
