from mmcls.models.losses import Accuracy
import math
from ..builder import HEADS, build_loss
from .base_head import BaseHead
import torch
import torch.nn.functional as F
from mmcls.models.losses import Accuracy
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcls.utils import GradReverse


@HEADS.register_module()
class SupConClsHead(BaseHead):

    def __init__(self,
                 in_channels,
                 num_classes,
                 mlp_dim,
                 supcon_loss=dict(type='SupConLoss', loss_weight=1.0),
                 cls_loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1,)):
        super(SupConClsHead, self).__init__()

        self.supcon_loss = build_loss(supcon_loss)
        self.cls_loss = build_loss(cls_loss)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim

        self._init_layers()
        self.init_weights()
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_accuracy = Accuracy(topk=self.topk)
      
        self.register_buffer('class_map', torch.zeros(num_classes, mlp_dim))

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.mlp_projector = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels, self.mlp_dim)
        )

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
        for m in self.mlp_projector:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

    def loss(self, features_mlp, cls_score, gt_label):
        losses = dict()

        loss_supcon = self.supcon_loss(features_mlp, gt_label)
        loss_cls = self.cls_loss(cls_score, gt_label.repeat(2))

        losses['supcon_loss'] = loss_supcon
        losses['cls_loss'] = loss_cls
        return losses

    def forward_train(self, features, gt_label=None):
        
        cls_scores = torch.cat(tuple([self.fc(feature) for feature in features]))
        features_mlp = [self.mlp_projector(feature) for feature in features]

        for label in range(31):
            label_num = 0
            label_features = torch.zeros_like(self.class_map[label, :])
            for idx, cat in enumerate(gt_label):
                if cat == label:
                    label_num += 2
                    label_features += features_mlp[0][idx, :].detach() + features_mlp[1][idx, :].detach()
            if label_num != 0:
                if sum(self.class_map[label, :]) == 0:
                    self.class_map[label, :] = label_features / label_num
                else:
                    self.class_map[label, :] = self.class_map[label, :] * 0.1 + label_features / label_num * 0.9
        features_mlp = torch.cat(tuple([feature.unsqueeze(1) for feature in features_mlp]), dim=1)

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


@HEADS.register_module()
class DASupConClsHead(BaseHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 mlp_dim,
                 sup_source_loss=None,
                 con_target_loss=None,
                 dist_loss=None,
                 w_loss=None,
                 cls_loss=None,
                 topk=(1, )):
        super(DASupConClsHead, self).__init__()

        self.sup_source_loss = None
        self.con_target_loss = None
        self.dist_loss = None
        self.cls_loss = None
        self.w_loss = None

        if sup_source_loss is not None:
            self.sup_source_loss = build_loss(sup_source_loss)
        if con_target_loss is not None:
            self.con_target_loss = build_loss(con_target_loss)
        if dist_loss is not None:
            self.dist_loss = build_loss(dist_loss)
        if cls_loss is not None:
            self.cls_loss = build_loss(cls_loss)
        if w_loss is not None:
            self.w_loss = build_loss(w_loss)
            self.gradreverse = GradReverse(1)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.mlp_dim = mlp_dim

        self._init_layers()
        self.init_weights()

        self.register_buffer('class_map', torch.zeros(self.num_classes, mlp_dim))
        self.register_buffer('class_map_t', torch.zeros(self.num_classes, mlp_dim))

        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_accuracy = Accuracy(topk=self.topk)


    def _init_layers(self):
        self.contrastive_projector = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels, self.mlp_dim))

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        """
        if self.w_loss is not None:
            self.w_projector = nn.Sequential(
                nn.Linear(self.mlp_dim, self.mlp_dim),
                nn.Linear(self.mlp_dim, self.mlp_dim),
                nn.Linear(self.mlp_dim, 1))
        """
    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
        for m in self.contrastive_projector:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
        """
        if w_loss is not None:
            for m in self.w_projector:
                if isinstance(m, nn.Linear):
                    normal_init(m, mean=0, std=0.01, bias=0)
        """
    def loss(self,
             mlp_source,
             mlp_target,
             cls_source,
             cls_target,
             source_label,
             target_label,
             reverse_source=None,
             reverse_target=None):

        batchsize = int(cls_source.shape[0]/2)

        losses = dict()

        if self.dist_loss is not None:
            losses['class_dist_target_loss'] = self.dist_loss(target_features, class_map)
       
        if self.w_loss is not None:
            losses['wdist_loss'] = self.w_loss(reverse_source, reverse_target)

        #_, pred_s = features_source.detach().topk(1,dim=1)
        #pred_s = pred_s.squeeze()
        #pred_s = pred_s.split(int(pred_s.shape[0]/2))
        #losses['class_acc_s[0]'] = sum(pred_s[0] == gt_label)/len(gt_label)
        #losses['class_acc_s[1]'] = sum(pred_s[1] == gt_label)/len(gt_label)
        #losses['class_dist_align'] = sum(pred[0] == target_pred[0])/len(target)
        #losses['correct_align'] = sum(torch.logical_and(target_pred[0] == target, pred[0] == target))/len(target)

        #source_pred = dist_max_idx_source.detach().squeeze()
        #source_pred = source_pred.split(int(source_pred.shape[0]/2))
        #losses['dist_acc[0]'] = sum(source_pred[0] == gt_label)/len(gt_label)
        
  
        """class prototype alignmenti evaluation"""
        #class_dist = torch.matmul(class_map, class_map_target.T).detach()
        #_, max_label = torch.max(class_dist, dim=1)
        #label_map = torch.arange(start=0, end=31, step=1).to(torch.device('cuda'))
        #losses['correct_count'] = torch.tensor(len(torch.where(max_label.squeeze() - label_map == 0)[0]), dtype=float)

        # Target element label
        #gt_combine_label = torch.cat((gt_label, target))
        #supcon_label = torch.cat((gt_label, target_label))

        """Loss Type"""
        #Type 1: concat source and target
        #features_mlp = torch.cat((features_mlp_source, features_mlp_target), dim=0) 
        #losses['supcon_combine_loss'] = self.supcon_loss(features_mlp.detach(), supcon_label)

        #features_mlp_test = features_mlp.clone().detach()
        #losses['supcon_combine_refer'] = self.supcon_loss(features_mlp.detach(), gt_combine_label)

        #Type 2: source and target seperate

        if self.con_target_loss is not None:
            features_mlp_target = torch.cat(mlp_target[0: batchsize].unsqueeze(1),
                                            mlp_target[batchsize:-1].unsqueeze(1), dim=1)
            target_label = torch.arange(target_label.shape[0]).to(torch.device('cuda'))
            losses['supcon_target_loss'] = self.con_target_loss(features_mlp_target, target_label)
            #losses['supcon_target_refer'] = self.supcon_loss(features_mlp_target.detach(), target)

        if self.sup_source_loss is not None:
            features_mlp_source = torch.cat((mlp_source[0: batchsize,:].unsqueeze(1),
                                            mlp_source[batchsize:,:].unsqueeze(1)), dim=1)
            losses['supcon_source_loss'] = self.sup_source_loss(features_mlp_source, source_label)

        #classification loss
        if self.cls_loss is not None:
            source_cls_label = source_label.repeat(2)
            #target_cls_label = target.repeat(2)
            #losses['target_cls_loss'] = self.cls_loss(features_target, target_cls_label)
            losses['source_cls_loss'] = self.cls_loss(cls_source, source_cls_label)

        #acc = self.compute_accuracy(features_source, source_cls_label)
        #assert len(acc) == len(self.topk)
        #losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

        return losses

    def forward_train(self, features_source, features_target, source_label=None, target_label=None):
        cls_source = self.fc(features_source)
        cls_target = self.fc(features_target)

        mlp_source = self.contrastive_projector(features_source)
        mlp_target = self.contrastive_projector(features_target)

        mlp_source = mlp_source / mlp_source.norm(dim=1, keepdim=True)
        mlp_target = mlp_target / mlp_target.norm(dim=1, keepdim=True)

        reverse_source = None
        reverse_target = None

        if self.w_loss is not None:
            reverse_source = self.gradreverse.apply(features_source)
            reverse_target = self.gradreverse.apply(features_target)

            reverse_source = self.contrastive_projector(reverse_source)
            reverse_target = self.contrastive_projector(reverse_target)

            reverse_source = reverse_source / reverse_source.norm(dim=1, keepdim=True)
            reverse_target = reverse_target / reverse_target.norm(dim=1, keepdim=True)

        #self.accumulate_map(mlp_source, mlp_target, source_label, target_label)

        losses = self.loss(mlp_source,
                           mlp_target, 
                           cls_source,
                           cls_target,
                           source_label,
                           target_label,
                           reverse_source,
                           reverse_target)

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        #cls_dist = torch.matmul(img_mlp, class_map.T)
        #pred = F.softmax(cls_dist, dim=1)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def accumulate_map(self, feat_s, feat_t, label_s, label_t):
        momentum = 0.1
        bs_size_s = int(feat_s.shape[0]/2)
        bs_size_t = int(feat_t.shape[0]/2)

        for label in range(self.num_classes):
            label_num = 0
            label_features = torch.zeros_like(self.class_map[label, :])
            label_num_t = 0
            label_features_t = torch.zeros_like(self.class_map_t[label, :])
            for idx, cat in enumerate(label_s):
                if cat == label:
                    label_num += 2
                    label_features += feat_s[idx, :].detach() + feat_s[idx + bs_size_s, :].detach()
            if label_num != 0:
                if sum(self.class_map[label, :]) == 0:
                    self.class_map[label, :] = label_features / label_num
                else:
                    self.class_map[label, :] = self.class_map[label, :] * (1 - momentum) \
                                               + label_features / label_num * momentum

            for idx, cat in enumerate(label_t):
                if cat == label:
                    label_num_t += 2
                    label_features_t += feat_t[idx, :].detach() + feat_t[idx + bs_size_s, :].detach()
            if label_num != 0:
                if sum(self.class_map_t[label, :]) == 0:
                    self.class_map_t[label, :] = label_features_t / label_num_t
                else:
                    self.class_map_t[label, :] = self.class_map_t[label, :] * (1 - momentum) \
                                               + label_features_t / label_num_t * momentum

