import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .da_base import DABaseClassifier


@CLASSIFIERS.register_module()
class SupConClsClassifier(DABaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None, feat_dim=128, class_num=31):
        super(SupConClsClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.feat_dim = feat_dim
        self.class_num = class_num

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feat_dim)
        )

        self.init_weights(pretrained=pretrained)
        self.class_map = torch.zeros(class_num, feat_dim).to(torch.device('cuda'))
        self.class_map_t = torch.zeros(class_num, feat_dim).to(torch.device('cuda'))

    def init_weights(self, pretrained=None):
        super(SupConClsClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
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
                img_split = self.fc(img_split)
                x.append(img_split)
        else:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
        return x

    def forward_train(self, img_s, img_t, gt_label_s, gt_label_t, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feat_s = [self.extract_feat(img_s[i]) for i in range(len(img_s))]
        feat_t = [self.extract_feat(img_t[i]) for i in range(len(img_t))]
        #feat_s = [feat_s[i] / feat_s[i].norm(dim=1).view(-1,1).repeat(1, feat_s[i].shape[1]) for i in range(len(feat_s))]
        #feat_t = [feat_t[i] / feat_t[i].norm(dim=1).view(-1,1).repeat(1, feat_t[i].shape[1]) for i in range(len(feat_t))]
        #x_s = [self.fc(feat_s[i]) / self.fc(feat_s[i]).norm(dim=1).view(-1, 1).repeat(1, self.fc(feat_s[i]).shape[1]) for i in range(len(feat_s))]
        #x_t = [self.fc(feat_t[i]) / self.fc(feat_t[i]).norm(dim=1).view(-1, 1).repeat(1, self.fc(feat_t[i]).shape[1]) for i in range(len(feat_t))]

        x_s = [self.fc(feat_s[i]) for i in range(len(feat_s))]
        x_t = [self.fc(feat_t[i]) for i in range(len(feat_t))]
        
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
                self.class_map[label, :] = self.class_map[label, :] * 0.1 + label_features / label_num * 0.9
            for idx, cat in enumerate(gt_label_t):
                if cat == label:
                    label_num_t += 2
                    label_features_t += x_t[0][idx, :].detach() + x_t[1][idx, :].detach()
            if label_num_t != 0:
                self.class_map_t[label, :] = self.class_map_t[label, :] * 0.5 + label_features_t / label_num_t * 0.5

#        if isinstance(x_s, list):
        x_s = torch.cat([x_s[0].unsqueeze(1), x_s[1].unsqueeze(1)], dim=1)
        x_t = torch.cat([x_t[0].unsqueeze(1), x_t[1].unsqueeze(1)], dim=1)
        #x = torch.cat((x_s, x_t), dim=0)

        losses = dict()
        loss = self.head.forward_train(x_s, x_t, torch.cat((feat_t[0], feat_t[1])), torch.cat((feat_s[0], feat_s[1])), gt_label_s, gt_label_t, self.class_map, self.class_map_t)
        losses.update(loss)

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)
