import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import cv2
import mmcv
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv import color_val
from mmcv.utils import print_log


class DABaseClassifier(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super(DABaseClassifier, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log(f'load model from: {pretrained}', logger='root')

    def forward_test(self, imgs, test_mode, **kwargs):
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(imgs) == 1:
            return self.simple_test(imgs[0], test_mode, **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')

    def forward(self, img_s, gt_label_s=None, img_t=None, gt_label_t=None, return_loss=True, test_mode='distance', **kwargs):
        if return_loss:
            return self.forward_train(img_s, gt_label_s, img_t, gt_label_t, **kwargs)
        else:
            return self.forward_test(img_s, test_mode=test_mode, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data_s, data_t=None, optimizer=None, **kwargs):

        if data_t is not None:
            losses = self(**data_s, **data_t)
        else:
            losses = self(**data_s)

        loss, log_vars = self._parse_losses(losses)

        if isinstance(data_s['img_s'], list):
            if data_t is None:
                samples = len(data_s['img_s'][0].data)
            else:
                samples = len(data_s['img_s'][0].data) + len(data_t['img_t'][0].data)
        else:
            samples = len(data_s['img_s'].data)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=samples)

        return outputs

    def val_step(self, data, optimizer):

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

