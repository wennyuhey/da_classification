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
        self.temp=0.1

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

    def loss(self, features_mlp_source, features_mlp_target, features_target, features_source, gt_label, target, class_map):
        losses = dict()
        target_features = torch.cat(torch.unbind(features_mlp_target, dim=1), dim=0)
        class_dot_dist = torch.div(torch.matmul(target_features, class_map.T), self.temp)
        dist_max, dist_max_idx = torch.max(class_dot_dist, dim=1, keepdim=True)
        exp_dist = torch.exp(class_dot_dist)
        log_class_prob = dist_max - torch.log(exp_dist.sum(1, keepdim=True))
        losses['class_dist_target'] = - log_class_prob.mean()
        #exp_dist = torch.exp(class_dot_dist)
        #losses['max_class_loss'], _ = torch.max(class_dot_dist, dim=1, keepdim=True)
        
   
        # Target element label
        target_label = torch.arange(start = 100, end = 100 + target.shape[0], step = 1).to(torch.device('cuda'))
        gt_combine_label = torch.cat((gt_label, target))
        supcon_label = torch.cat((gt_label, target_label))

        """Loss Type"""
        #Type 1: concat source and target
        features_mlp = torch.cat((features_mlp_source, features_mlp_target), dim=0) 
        losses['supcon_combine_loss'] = self.supcon_loss(features_mlp.detach(), supcon_label)

        features_mlp_test = features_mlp.clone().detach()
        losses['supcon_combine_refer'] = self.supcon_loss(features_mlp_test, gt_combine_label)

        #Type 2: source and target seperate
        losses['supcon_target_loss'] = self.supcon_loss(features_mlp_target, target_label)
        losses['supcon_target_refer'] = self.supcon_loss(features_mlp_target.detach(), target)

        losses['supcon_source_loss'] = self.supcon_loss(features_mlp_source, gt_label)

        #classification loss
        target_cls_label = target.repeat(2)
        source_cls_label = gt_label.repeat(2)

        losses['target_cls_loss'] = self.cls_loss(features_target.detach(), target_cls_label)
        losses['source_cls_loss'] = self.cls_loss(features_source, source_cls_label)

        #acc = self.compute_accuracy(features_source, source_cls_label)
        #assert len(acc) == len(self.topk)
        #losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

        return losses

    def forward_train(self, features_mlp_source, features_mlp_target, features_target, features, gt_label=None, target_label=None, class_map=None):
        cls_scores = self.fc(features)
        target_cls_scores = self.fc(features_target)
        losses = self.loss(features_mlp_source, features_mlp_target, target_cls_scores, cls_scores, gt_label, target_label, class_map)
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

