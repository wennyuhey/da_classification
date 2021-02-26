from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead
import torch
import torch.nn.functional as F
from mmcls.models.losses import Accuracy
import torch.nn as nn
from mmcv.cnn import normal_init


@HEADS.register_module()
class DisAlignHead(BaseHead):
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

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
        """
        normal_init(self.projector1, mean=0, std=0.01, bias=0)
        normal_init(self.projector2, mean=0, std=0.01, bias=0)
        """

    def loss(self, features_mlp_source, features_mlp_target, features_target, features_source, gt_label, target, class_map, class_map_target):
        losses = dict()
        source_features = torch.cat(torch.unbind(features_mlp_source, dim=1), dim=0)
        target_features = torch.cat(torch.unbind(features_mlp_target, dim=1), dim=0)

        """Distance between target features and source class prototype"""
        class_dot_dist = torch.div(torch.matmul(target_features, class_map.T), self.temp)
        dist_max, dist_max_idx = torch.max(class_dot_dist, dim=1, keepdim=True)
        class_dot_dist = class_dot_dist - dist_max
        exp_dist = torch.exp(class_dot_dist)
        log_class_prob = - torch.log(exp_dist.sum(1, keepdim=True))
        #losses['class_dist_target'] = - log_class_prob.mean()

        """
        target_pred = dist_max_idx.detach().squeeze()
        target_pred = target_pred.split(int(target_pred.shape[0]/2))


        losses['dist_acc[0]'] = sum(target_pred[0] == target)/len(target)
        #losses['dist_acc[1]'] = sum(target_pred[1] == target)/len(target)
        
        _, pred = features_target.detach().topk(1, dim=1)
        pred = pred.squeeze()
        pred = pred.split(int(pred.shape[0]/2))
        losses['class_acc[0]'] = sum(pred[0] == target)/len(target)
        #losses['class_acc[1]'] = sum(pred[1] == target)/len(target)
        """
        _, pred_s = features_source.detach().topk(1,dim=1)
        pred_s = pred_s.squeeze()
        pred_s = pred_s.split(int(pred_s.shape[0]/2))
        losses['class_acc_s[0]'] = sum(pred_s[0] == gt_label)/len(gt_label)
        #losses['class_acc_s[1]'] = sum(pred_s[1] == gt_label)/len(gt_label)
        #losses['class_dist_align'] = sum(pred[0] == target_pred[0])/len(target)
        #losses['correct_align'] = sum(torch.logical_and(target_pred[0] == target, pred[0] == target))/len(target)
        
        class_dot_dist_source = torch.div(torch.matmul(source_features.detach(), class_map.T), self.temp)
        dist_max_source, dist_max_idx_source = torch.max(class_dot_dist_source, dim=1, keepdim=True)
        #class_dot_dist_source = class_dot_dist_source - dist_max_source
        #exp_dist_source = torch.exp(class_dot_dist_source)
        #log_class_prob_source = - torch.log(exp_dist_source.sum(1, keepdim=True))
        #losses['class_dist_source'] = - log_class_prob_source.mean()

        source_pred = dist_max_idx_source.detach().squeeze()
        source_pred = source_pred.split(int(source_pred.shape[0]/2))
        losses['dist_acc[0]'] = sum(source_pred[0] == gt_label)/len(gt_label)
        
  
        """class prototype alignmenti evaluation"""
        class_dist = torch.matmul(class_map, class_map_target.T).detach()
        _, max_label = torch.max(class_dist, dim=1)
        label_map = torch.arange(start=0, end=31, step=1).to(torch.device('cuda'))
        #losses['correct_count'] = torch.tensor(len(torch.where(max_label.squeeze() - label_map == 0)[0]), dtype=float)

        """
        class_dot_dist_s = torch.div(torch.matmul(source_features, class_map.T), self.temp)
        dist_max_s, _ = torch.max(class_dot_dist_s, dim=1, keepdim=True)
        class_dot_dist = class_dot_dist_s - dist_max_s
        exp_dist_s = torch.exp(class_dot_dist)
        log_class_prob_s = - torch.log(exp_dist_s.sum(1, keepdim=True))
        losses['max_class_source'] = - log_class_prob_s.mean().detach()
        """
        
        # Target element label
        target_label = torch.arange(start = 100, end = 100 + target.shape[0], step = 1).to(torch.device('cuda'))
        gt_combine_label = torch.cat((gt_label, target))
        supcon_label = torch.cat((gt_label, target_label))

        """Loss Type"""
        #Type 1: concat source and target
        #features_mlp = torch.cat((features_mlp_source, features_mlp_target), dim=0) 
        #losses['supcon_combine_loss'] = self.supcon_loss(features_mlp.detach(), supcon_label)

        #features_mlp_test = features_mlp.clone().detach()
        #losses['supcon_combine_refer'] = self.supcon_loss(features_mlp.detach(), gt_combine_label)

        #Type 2: source and target seperate
#        losses['supcon_target_loss'] = self.supcon_loss(features_mlp_target, target_label)
        #losses['supcon_target_refer'] = self.supcon_loss(features_mlp_target.detach(), target)
        losses['supcon_source_loss'] = self.supcon_loss(features_mlp_source, gt_label)

        #classification loss
        target_cls_label = target.repeat(2)
        source_cls_label = gt_label.repeat(2)

        #losses['target_cls_loss'] = self.cls_loss(features_target, target_cls_label)
        losses['source_cls_loss'] = self.cls_loss(features_source, source_cls_label)

        #acc = self.compute_accuracy(features_source, source_cls_label)
        #assert len(acc) == len(self.topk)
        #losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

        return losses

    def forward_train(self, features_mlp_source, features_mlp_target, features_target, features, gt_label=None, target_label=None, class_map=None, class_map_t=None):
        cls_scores = self.fc(features)
        target_cls_scores = self.fc(features_target)
        losses = self.loss(features_mlp_source, features_mlp_target, target_cls_scores, cls_scores, gt_label, target_label, class_map, class_map_t)
        return losses

    def simple_test(self, img, img_mlp, class_map):
        """Test without augmentation."""
        #cls_score = self.fc(img)
        cls_dist = torch.matmul(img_mlp, class_map.T)
        pred = F.softmax(cls_dist, dim=1)
        """
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        """
        pred = list(pred.detach().cpu().numpy())
        return pred

