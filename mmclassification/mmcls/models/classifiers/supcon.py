import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class SupConClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None, mlp_dim=128):
        super(SupConClassifier, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)
        """
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, mlp_dim)
        )
        """
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SupConClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        """
        for m in self.projector:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
        """
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        if isinstance(img, list):
            x = []
            for img_split in img:
                img_split = self.backbone(img_split)
                if self.with_neck:
                    img_split = self.neck(img_split)
                x.append(img_split)
        else:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)
