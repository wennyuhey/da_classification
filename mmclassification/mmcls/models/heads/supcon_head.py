from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead
import torch
import torch.nn.functional as F
from mmcls.models.losses import Accuracy
import torch.nn as nn
from mmcv.cnn import normal_init


@HEADS.register_module()
class SupConHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='SupConLoss', loss_weight=1.0)):
        super(SupConHead, self).__init__()
        assert isinstance(loss, dict)
        self.compute_loss = build_loss(loss)

    def loss(self, cls_score, gt_label):
        losses = dict()
        loss = self.compute_loss(cls_score, gt_label)
        losses['loss'] = loss
        return losses

    def forward_train(self, features, gt_label=None):
        losses = self.loss(features, gt_label)
        return losses

@HEADS.register_module()
class SupConClsHead(BaseHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 sup_loss=dict(type='SupConLoss', loss_weight=1.0),
                 cls_loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(SupConClsHead, self).__init__()
        self.supcon_loss = build_loss(sup_loss)
        self.cls_loss = build_loss(cls_loss)
        self.num_classes = num_classes
        self.in_channels = in_channels

        self._init_layers()
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_accuracy = Accuracy(topk=self.topk)


    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        """
        self.relu = nn.ReLU()
        self.projector1 = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.projector2 = nn.Linear(self.in_channels, 64, bias=False)
        self.fc = nn.Linear(64, self.num_classes)
        """

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
        """
        normal_init(self.projector1, mean=0, std=0.01, bias=0)
        normal_init(self.projector2, mean=0, std=0.01, bias=0)
        """

    def loss(self, features_mlp, features, gt_label):
        losses = dict()
        target_label = torch.arange(start = 100, end = 100 + features_mlp.shape[0] - gt_label.shape[0], step = 1).to(torch.device('cuda'))
        supcon_label = torch.cat((gt_label, target_label))
        losses['supcon_loss'] = self.supcon_loss(features_mlp, supcon_label)
        cls_label = gt_label.repeat(2) 
        losses['cls_loss'] = self.cls_loss(features, cls_label)
        acc = self.compute_accuracy(features, cls_label)
        assert len(acc) == len(self.topk)
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

        return losses

    def forward_train(self, features_mlp, features, gt_label=None):
        cls_scores = self.fc(features)
        losses = self.loss(features_mlp, cls_scores, gt_label)
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

