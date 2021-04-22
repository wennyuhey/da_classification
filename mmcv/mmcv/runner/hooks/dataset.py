# Copyright (c) Open-MMLab. All rights reserved.
import torch

from .hook import HOOKS, Hook


@HOOKS.register_module()
class DatasetHook(Hook):

    def __init__(self, dataloader, interval=1):
        self.dataloader = dataloader

    def before_train_epoch(self, runner):
        if self.before_epoch and self.every_n_epochs(runner, self.interval):
           from mmcls.apis import da_single_gpu_test
           results = da_single_gpu_test(runner.model, self.dataloader, show=False)
           _, preds = torch.max(results, dim=1)
           runner.data_loader.dataset.updata(preds) 
