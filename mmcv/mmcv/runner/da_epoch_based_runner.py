# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from .da_base_runner import DABaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class DAEpochBasedRunner(DABaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_s, train_mode, data_t=None, **kwargs):
        data_s['img_s'] = data_s.pop('img')
        data_s['gt_label_s'] = data_s.pop('gt_label')
        if self.source_only is not True:
            data_t['img_t'] = data_t.pop('img')
            data_t['gt_label_t'] = data_t.pop('gt_label')
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_s, data_t, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_s, data_t, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_s, data_t, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader_s, data_loader_t=None, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader_s = data_loader_s
        self.iter_s = iter(self.data_loader_s)
        if not self.source_only:
            self.data_loader_t = data_loader_t
            self.iter_t = iter(self.data_loader_t)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        if self.source_only:
            for i, input_data_s in enumerate(self.data_loader_s):
                self._inner_iter = i
                self.call_hook('before_train_iter')
                if self.batch_processor is None:
                    self.run_iter(input_data_s, train_mode=True)
                    self.call_hook('after_train_iter')
                    self._iter += 1
        else:
            for i in range(self._max_iter_per_epoch):
                self._inner_iter = i
                self.call_hook('before_train_iter')
                if self.batch_processor is None:
                    try:
                        input_data_s = self.iter_s.__next__()
                    except:
                        self.iter_s = iter(self.data_loader_s)
                        input_data_s = self.iter_s.__next__()
                    try:
                        input_data_t = self.iter_t.__next__()
                    except:
                        self.iter_t = iter(self.data_loader_t)
                        input_data_t = self.iter_t.__next__()
                    self.run_iter(input_data_s, True, input_data_t)
                    self.call_hook('after_train_iter')
                    self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders_s, data_loaders_t, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders_s, list)
        #assert isinstance(data_loaders_t, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders_s) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iter_per_epoch = len(data_loaders_s[i]) if self.source_only else max(len(data_loaders_s[i]), len(data_loaders_t[i]))
                self._max_iters = self._max_epochs * self._max_iter_per_epoch
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    if self.source_only:
                        epoch_runner(data_loaders_s[i])
                    else:
                        epoch_runner(data_loaders_s[i], data_loaders_t[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class DARunner(DAEpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
