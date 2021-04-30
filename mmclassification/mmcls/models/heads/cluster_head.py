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
class DASupClusterHead(BaseHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 mlp_dim,
                 momentum=0.9,
                 threshold=0.8,
                 sup_source_loss=None,
                 combined_loss=None,
                 con_target_loss=None,
                 soft_ce=None,
                 cls_loss=None,
                 select_feat=None,
                 topk=(1, ),
                 frozen_map=True,
                 domain_loss=None,
                 mlp_cls=True):
        super(DASupClusterHead, self).__init__()

        self.sup_source_loss = None
        self.con_target_loss = None
        self.cls_loss = None
        self.frozen_map = frozen_map
        self.momentum = momentum
        self.domain_loss = None
        self.combined_loss = None
        self.select_feat = None
        self.threshold = threshold
        self.soft_cls = None
        self.mlp_flag = mlp_cls
        self.epsilon = 0.05

        if sup_source_loss is not None:
            self.sup_source_loss = build_loss(sup_source_loss)
        if con_target_loss is not None:
            self.con_target_loss = build_loss(con_target_loss)
        if cls_loss is not None:
            self.cls_loss = build_loss(cls_loss)
        if domain_loss is not None:
            self.domain_loss = build_loss(domain_loss)
            self.gradreverse = GradReverse(1)
        if select_feat is not None:
            self.feat_select = build_loss(select_feat)
        if soft_ce is not None:
            self.soft_cls = build_loss(soft_ce)
        if combined_loss is not None:
            self.combined_loss = build_loss(combined_loss)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.mlp_dim = mlp_dim

        self._init_layers()
        self.init_weights()

        #self.register_buffer('class_map', torch.zeros(self.num_classes, mlp_dim))
        self.register_buffer('class_map_verse', torch.zeros(self.num_classes, mlp_dim))

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

        if self.mlp_flag:
            self.fc = nn.Linear(self.mlp_dim, self.num_classes)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.class_map = nn.Parameter(torch.FloatTensor(self.num_classes, self.mlp_dim).normal_(mean=0, std=0.01))

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
        for m in self.contrastive_projector:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

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


        if self.domain_loss is not None:
            losses['domain_loss']= self.domain_loss(reverse_source, reverse_target)

        if self.con_target_loss is not None:
            features_mlp_target = torch.cat((mlp_target[0: batchsize].unsqueeze(1),
                                            mlp_target[batchsize: batchsize*2].unsqueeze(1)), dim=1)
            target_label = torch.arange(target_label.shape[0]).to(torch.device('cuda'))
            losses['supcon_target_loss'] = self.con_target_loss(features_mlp_target, target_label)
            #losses['supcon_target_refer'] = self.supcon_loss(features_mlp_target.detach(), target)

        if self.sup_source_loss is not None:
            features_mlp_source = torch.cat((mlp_source[0: batchsize,:].unsqueeze(1),
                                            mlp_source[batchsize:,:].unsqueeze(1)), dim=1)
            losses['supcon_source_loss'] = self.sup_source_loss(features_mlp_source, source_label)

        if self.combined_loss is not None:
            mlp_t_top = mlp_target[0: batchsize, :]
            mlp_t_bot = mlp_target[batchsize: , :]
            
            dist_top = torch.matmul(mlp_t_top, self.class_map.T)
            dist_bot = torch.matmul(mlp_t_bot, self.class_map.T)

            cls_t, idx_t = torch.max(dist_top, dim=1)
            cls_b, idx_b = torch.max(dist_bot, dim=1)

            cls_mask = cls_t > cls_b
            idx = idx_t * cls_mask + idx_b * ~cls_mask
            cls = torch.maximum(cls_t, cls_b)

            selected_idx = torch.where(cls > self.threshold)[0]
            unselected_idx = torch.where(cls < 4.7)[0]

            target_selected_t = torch.index_select(mlp_target[0:batchsize], 0, selected_idx)
            target_selected_b = torch.index_select(mlp_target[batchsize:,:], 0, selected_idx)
            label_t = torch.index_select(idx, 0, selected_idx)
            features_mlp = torch.cat((torch.cat((mlp_source[0: batchsize,:], target_selected_t)).unsqueeze(1),
                                     torch.cat((mlp_source[batchsize:,:], target_selected_b)).unsqueeze(1)), dim=1)
            label_combine = torch.cat((source_label, label_t))
            losses['combined_supcon_loss'] = self.combined_loss(features_mlp, label_combine)

        if self.soft_cls is not None:
            if len(unselected_idx) != 0:
                target_cls_t = torch.index_select(cls_target[0:batchsize, :], 0, unselected_idx)
                target_cls_t = F.softmax(target_cls_t, dim=1)
                label_uncertain_t = torch.index_select(idx_t, 0, unselected_idx)
                target_cls_b = torch.index_select(cls_target[batchsize:, :], 0, unselected_idx)
                target_cls_b = F.softmax(target_cls_b, dim=1)
                label_uncertain_b = torch.index_select(idx_b, 0, unselected_idx)
                class_map_t = self.class_map[label_uncertain_t]
                class_map_b = self.class_map[label_uncertain_b]
                #cls_map_t = F.softmax(self.fc(class_map_t), dim=1)
                #cls_map_b = F.softmax(self.fc(class_map_b), dim=1)
                cls_map_t = self.fc(class_map_t)
                cls_map_b = self.fc(class_map_b)
                losses['target_map_ce'] = (self.soft_cls(target_cls_t, cls_map_b.detach()) + self.soft_cls(target_cls_b, cls_map_t.detach())) / 2

        dist_target = torch.matmul(mlp_target, self.class_map.T)
        Q = self.sinkhorn_knopp(dist_target.detach())
        losses['target_cls_loss'] = -torch.mean(torch.sum(Q * F.log_softmax(dist_target, dim=1), dim=1))
       
        if self.cls_loss is not None:
            if(source_label.size(0) != cls_source.size(0)):
                source_cls_label = source_label.repeat(2)
            else:
                source_cls_label = source_label
            #losses['target_cls_loss'] = self.cls_loss(features_target, target_cls_label)
            #losses['source_cls_loss'] = self.cls_loss(cls_source, source_cls_label)
            dist_source = torch.matmul(mlp_source, self.class_map.T)
            source_score = F.softmax(dist_source, dim=1)
            losses['source_cls_loss'] = self.cls_loss(source_score, source_cls_label)

        return losses

    def forward_train(self, features_source, features_target, source_label=None, target_label=None):
        
        mlp_source = self.contrastive_projector(features_source)
        mlp_source = mlp_source / mlp_source.norm(dim=1, keepdim=True)

        if self.mlp_flag:
            cls_source = self.fc(mlp_source)
        else:
            cls_source = self.fc(features_source)

        if features_target is not None:
            mlp_target = self.contrastive_projector(features_target)
            mlp_target = mlp_target / mlp_target.norm(dim=1, keepdim=True)

            if self.mlp_flag:
                cls_target = self.fc(mlp_target)
            else:
                cls_target = self.fc(features_target)
        else:
            cls_target = None
            mlp_target = None

        reverse_source = None
        reverse_target = None

        if self.domain_loss is not None:
            reverse_source = self.gradreverse.apply(features_source)
            reverse_target = self.gradreverse.apply(features_target)
        """
        if self.frozen_map is False:
            self.accumulate_map(mlp_source, mlp_target, source_label, target_label)
        """
        losses = self.loss(mlp_source,
                           mlp_target, 
                           cls_source,
                           cls_target,
                           source_label,
                           target_label,
                           reverse_source,
                           reverse_target)

        return losses

    def distance_test(self, img):
        """Test without augmentation."""
        img_mlp = self.contrastive_projector(img)
        cls_dist = torch.matmul(img_mlp, self.class_map.T)
        pred = F.softmax(cls_dist, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return img, img_mlp, pred

    def fc_test(self, img):
        if self.mlp_flag:
            img_mlp = self.contrastive_projector(img)
            cls_score = self.fc(img_mlp)
        else:
            cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def accumulate_map(self, feat_s, feat_t, label_s, label_t):
        bs_size_s = int(feat_s.shape[0]/2)
        bs_size_t = int(feat_t.shape[0]/2)
        refs = torch.LongTensor(range(self.num_classes)).unsqueeze(1).to(torch.device('cuda'))
        gt = label_s.repeat(2)
        mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        num_mask = mask.sum(dim=1)
        logic_mask = num_mask == 0
        map_mask = (self.class_map.sum(dim=1) == 0).reshape(-1, 1)
        feature = feat_s.detach().unsqueeze(0)
        update_feature = torch.sum(feature * mask, dim=1) / (num_mask + 10e-6)
        self.class_map = map_mask * update_feature + \
                         ~map_mask * (logic_mask * self.class_map + ~logic_mask*(self.class_map * self.momentum + update_feature * (1 - self.momentum)))
#        self.class_map = self.class_map/self.class_map.norm(dim=1, keepdim=True)
      
#        self.class_map = self.class_map.detach()
#        self.class_map_verse = self.class_map_verse * self.momentum + torch.sum(feature * mask, dim=1) * (1 - self.momentum)

    def sinkhorn_knopp(self, dist):
        Q = torch.exp(dist/self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]
        Q_sum = torch.sum(Q)
        Q /= Q_sum

        #for i in range(self.sinkhorn_iterations):
        for i in range(3):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            sum_of_cols = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_cols
            Q /= B
    
        Q *= B

        return Q.t()
