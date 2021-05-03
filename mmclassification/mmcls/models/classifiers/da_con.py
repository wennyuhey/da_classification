import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck, build_loss
from .da_base import DABaseClassifier
import copy
from mmcls.utils import GradReverse


@CLASSIFIERS.register_module()
class DASupConClsClassifier(DABaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(DASupConClsClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.with_wloss = False

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.aux = ('Aux' in backbone.type)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(DASupConClsClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img, domain):
        """Directly extract features from the backbone + neck
        """
        if isinstance(img, list):
            x = []
            for img_split in img:
                if self.aux is True:
                    img_split = self.backbone(img_split, domain)
                else:
                    img_split = self.backbone(img_split)
                if self.with_neck:
                    img_split = self.neck(img_split)
                x.append(img_split)
        else:
            if self.aux is True:
                x = self.backbone(img, domain)
            else:
                x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
        return x

    def forward_train(self, img_s, gt_label_s, img_t=None, gt_label_t=None, **kwargs):

        feat_t = None
        if isinstance(img_s, list):
            feat_s = [self.extract_feat(img_s[i], torch.tensor([0])) for i in range(len(img_s))]
            feat_s = torch.cat(tuple(feat_s))
            if img_t is not None:
                feat_t = [self.extract_feat(img_t[i], torch.tensor([1])) for i in range(len(img_t))]
                feat_t = torch.cat(tuple(feat_t))
        else:
            feat_s = self.extract_feat(img_s, torch.tensor([0]))
            if img_t is not None:
                feat_t = self.extract_feat(img_t, torch.tensor([1]))

        losses = dict()
        loss = self.head.forward_train(feat_s, feat_t, gt_label_s, gt_label_t, **kwargs)
        losses.update(loss)

        return losses

    def simple_test(self, img, domain, test_mode='distance'):
        """Test without augmentation."""
        x = self.extract_feat(img, domain)
        if test_mode == 'fc':
            return self.head.fc_test(x)
        elif test_mode == 'distance':
            return self.head.distance_test(x)
        else:
            raise ValueError('Test mode {} is not supported'.format(test_mode))
