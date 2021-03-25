import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck, build_loss
from .da_base import DABaseClassifier
import copy
from mmcls.utils import GradReverse


@CLASSIFIERS.register_module()
class DASupConClsClassifier(DABaseClassifier):

    def __init__(self, backbone, neck=None, head=None, wloss=None, pretrained=None):
        super(DASupConClsClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.with_wloss = False

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        if wloss is not None:
            self.wloss = build_loss(wloss)
            self.with_wloss = True
            self.gradreverse = GradReverse(1)

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
                img_split = self.backbone(img_split, domain)
                if self.with_neck:
                    img_split = self.neck(img_split)
                img_split = self.fc(img_split)
                x.append(img_split)
        else:
            x = self.backbone(img, domain)
            if self.with_neck:
                x = self.neck(x)
        return x

    def forward_train(self, img_s, img_t, gt_label_s, gt_label_t, **kwargs):
        feat_s = [self.extract_feat(img_s[i], torch.tensor([0])) for i in range(len(img_s))]
        feat_t = [self.extract_feat(img_t[i], torch.tensor([1])) for i in range(len(img_t))]

        #x_s = [self.fc(feat_s[i]) for i in range(len(feat_s))]
        #x_t = [self.fc(feat_t[i]) for i in range(len(feat_t))]
        
        #x_s = [x/x.norm(dim=1, keepdim=True) for x in x_s]
        #x_t = [x/x.norm(dim=1, keepdim=True) for x in x_t]
        """
        for label in range(31):
            label_num = 0
            label_features = torch.zeros_like(self.class_map[label, :])
            label_num_t = 0
            label_features_t = torch.zeros_like(self.class_map[label, :])
            for idx, cat in enumerate(gt_label_s):
                if cat == label:
                    label_num += 2
                    label_features += x_s[0][idx, :].detach() + x_s[1][idx, :].detach()
            if label_num != 0:
                if sum(self.class_map[label, :]) == 0:
                    self.class_map[label, :] = label_features / label_num
                else:
                    self.class_map[label, :] = self.class_map[label, :] * 0.9 + label_features / label_num * 0.1
            
            for idx, cat in enumerate(gt_label_t):
                if cat == label:
                    label_num_t += 2
                    label_features_t += x_t[0][idx, :].detach() + x_t[1][idx, :].detach()
            if label_num_t != 0:
                if sum(self.class_map_t[label, :]) == 0:
                    self.class_map_t[label, :] = label_features_t / label_num_t
                else:
                    self.class_map_t[label, :] = self.class_map_t[label, :] * 0.9 + label_features_t / label_num_t * 0.1
        """

        #if isinstance(x_s, list):
        #x_s = torch.cat([x_s[0].unsqueeze(1), x_s[1].unsqueeze(1)], dim=1)
        #x_t = torch.cat([x_t[0].unsqueeze(1), x_t[1].unsqueeze(1)], dim=1)
        #x = torch.cat((x_s, x_t), dim=0)
        feat_s = torch.cat((feat_s[0], feat_s[1]))
        feat_t = torch.cat((feat_t[0], feat_t[1]))

        losses = dict()
        if self.with_wloss:
            
            feature_s = self.gradreverse.apply(feat_s)
            feature_t = self.gradreverse.apply(feat_t)
            feature_s = self.fc(feature_s)
            feature_t = self.fc(feature_t)
            feature_s = feature_s/feature_s.norm(dim=1, keepdim=True)
            feature_t = feature_t/feature_t.norm(dim=1, keepdim=True)
            loss_wdist = self.wloss(feature_s, feature_t)
            losses.update(loss_wdist)
        
        loss = self.head.forward_train(feat_s, feat_t, gt_label_s, gt_label_t)
        losses.update(loss)

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img, torch.tensor([1]))
        #x = self.fc(x)
        #x_mlp = self.fc(x)
        return self.head.simple_test(x)
