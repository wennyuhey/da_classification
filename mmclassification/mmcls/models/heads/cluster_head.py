from mmcls.models.losses import Accuracy
import math
from ..builder import HEADS, build_loss
from .base_head import BaseHead
import torch
import torch.nn.functional as F
from mmcls.models.losses import Accuracy
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.cnn.bricks import build_linear_layer
from mmcls.utils import GradReverse
import torch.distributed as distributed

@HEADS.register_module()
class DASupClusterHead(BaseHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 mlp_dim,
                 distributed=False,
                 oracle=False,
                 cluster=False,
                 momentum=0.9,
                 threshold=0.8,
                 epsilon=0.05,
                 sup_source_loss=None,
                 combined_loss=None,
                 con_target_loss=None,
                 cls_loss=None,
                 barlow_loss=False,
                 select_feat=None,
                 topk=(1, ),
                 pseudo=False,
                 frozen_map=True,
                 mlp_cls=True):
        super(DASupClusterHead, self).__init__()

        self.sup_source_loss = None
        self.con_target_loss = None
        self.cls_loss = None
        self.frozen_map = frozen_map
        self.momentum = momentum
        self.combined_loss = None
        self.threshold = threshold
        self.soft_cls = None
        self.barlow_loss = barlow_loss
        self.mlp_flag = mlp_cls
        self.epsilon = epsilon
        self.distributed = distributed
        self.oracle = oracle
        self.cluster = cluster
        self.pseudo = pseudo

        if sup_source_loss is not None:
            self.sup_source_loss = build_loss(sup_source_loss)
        if con_target_loss is not None:
            self.con_target_loss = build_loss(con_target_loss)
        if cls_loss is not None:
            self.cls_loss = build_loss(cls_loss)
        if combined_loss is not None:
            self.combined_loss = build_loss(combined_loss)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.mlp_dim = mlp_dim

        self._init_layers()
        self.init_weights()

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
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels, self.mlp_dim))

        if self.mlp_flag:
            self.fc = nn.Linear(self.mlp_dim, self.num_classes)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        
        self.class_map = build_linear_layer({'type':'NormLinear',
                                             'in_features': self.in_channels,
                                             'out_features': self.num_classes,
                                             'bias': False})
        self.mlp_class_map = build_linear_layer({'type':'NormLinear',
                                             'in_features': self.mlp_dim,
                                             'out_features': self.num_classes,
                                             'bias': False})

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
        for m in self.contrastive_projector:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def loss(self,
             features_source,
             features_target,
             mlp_source,
             mlp_target,
             cls_source,
             cls_target,
             source_label,
             target_label,
             target_pseudo,
             reverse_source=None,
             reverse_target=None):

        batchsize = len(source_label)
        losses = dict()
        if self.barlow_loss:
            scale_loss=0.5
            lambd = 0.01
            mlp_top = torch.cat((mlp_source[: batchsize,:], mlp_target[: batchsize, :]))
            mlp_bottom = torch.cat((mlp_source[batchsize:,:], mlp_target[batchsize:, :]))
            c = torch.matmul(mlp_top.T, mlp_bottom)
            c.div_(batchsize)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(scale_loss)
            off_diag = self.off_diagonal(c).pow_(2).sum().mul(scale_loss)
            losses['barlow_loss'] = on_diag + lambd * off_diag

        if self.con_target_loss is not None:
            features_mlp_target = torch.cat((mlp_target[0: batchsize].unsqueeze(1),
                                            mlp_target[batchsize: batchsize*2].unsqueeze(1)), dim=1)
            losses['supcon_target_loss'] = self.con_target_loss(features_mlp_target)

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

            target_selected_t = torch.index_select(mlp_target[0:batchsize], 0, selected_idx)
            target_selected_b = torch.index_select(mlp_target[batchsize:,:], 0, selected_idx)
            label_t = torch.index_select(idx, 0, selected_idx)
            features_mlp = torch.cat((torch.cat((mlp_source[0: batchsize,:], target_selected_t)).unsqueeze(1),
                                     torch.cat((mlp_source[batchsize:,:], target_selected_b)).unsqueeze(1)), dim=1)
            label_combine = torch.cat((source_label, label_t))
            losses['combined_supcon_loss'] = self.combined_loss(features_mlp, label_combine)

        if self.cluster:
            source_dist = self.class_map(features_source)
            losses['map_kl_loss'] = self.cls_loss(source_dist, source_label.repeat(self.times_source))
            mlp_source_dist = self.mlp_class_map(mlp_source)
            losses['mlp_kl_loss'] = self.cls_loss(mlp_source_dist, source_label.repeat(self.times_source))
            
            feat_t = features_target / features_target.norm(dim=1, keepdim=True)
            dist_target = self.class_map(feat_t)
            mlp_dist_target = self.mlp_class_map(mlp_target)

            if not self.oracle:
                dist_target = dist_target.reshape(self.times_target, batchsize, -1)
                C = torch.zeros_like(dist_target[0]).to(torch.device('cuda'))
                dist_max = torch.zeros((batchsize, 1)).to(torch.device('cuda'))
                #pred = torch.zeros((batchsize, 1)).to(torch.device('cuda')) - 1
                for i in range(self.times_target):
                    d = dist_target[i]
                    d_max, d_pred = torch.max(d, dim=1, keepdim=True)
                    mask = dist_max > d_max
                    C = C * mask + d * ~mask
                    #pred = pred * mask + d_pred * ~mask
                    dist_max = dist_max * mask + d_max * ~mask
                C = C - dist_max
                Q = self.sinkhorn_knopp(C.detach())
                Q = Q.repeat(self.times_target, 1)
                
                losses['target_map_loss'] = - torch.mean(torch.sum(Q * F.log_softmax(dist_target.reshape(batchsize * self.times_target, -1), dim=1), dim=1))
         
                """
                dist_mean = dist_max.mean()
                dist_std = dist_max.std()
                threshold = dist_mean - dist_std
                confuse_idx = torch.where(dist_max < threshold)[0]
                pseudo_label = pred[confuse_idx].squeeze().long()
                target_prob = F.softmax(cls_target[confuse_idx]/0.07, dim=1)
                losses['confuse_loss'] = F.cross_entropy(target_prob, pseudo_label)
                """
        
                mlp_dist_target = mlp_dist_target.reshape(self.times_target, batchsize, -1)
                mlp_C = torch.zeros_like(mlp_dist_target[0]).to(torch.device('cuda'))
                mlp_dist_max = torch.zeros((batchsize, 1)).to(torch.device('cuda'))
                for i in range(self.times_target):
                    d = mlp_dist_target[i]
                    d_max, _ = torch.max(d, dim=1, keepdim=True)
                    mask = mlp_dist_max > d_max
                    mlp_C = mlp_C * mask + d * ~mask
                    mlp_dist_max = mlp_dist_max * mask + d_max * ~mask
        
                mlp_Q = self.sinkhorn_knopp(mlp_C.detach())
                mlp_Q = mlp_Q.repeat(self.times_target, 1)
                losses['mlp_target_map_loss'] = - torch.mean(torch.sum(mlp_Q * F.log_softmax(mlp_dist_target.reshape(batchsize * self.times_target, -1), dim=1), dim=1))
               
        if self.cls_loss is not None:
            source_cls_label = source_label.repeat(self.times_source)
            target_cls_label = target_label.repeat(self.times_target)
            target_pseudo_label = target_pseudo.repeat(self.times_target)
            if self.oracle:
                dist_target = self.class_map(features_target)
                #losses['target_cls_loss'] = F.cross_entropy(cls_target, target_cls_label)
                losses['target_map_loss'] = F.cross_entropy(dist_target, target_cls_label)
                #losses['target_mlp_map_loss'] = F.cross_entropy(mlp_dist_target, target_cls_label)
            if self.pseudo and target_pseudo[0] != -1:
                losses['target_pseudo_loss'] = self.cls_loss(cls_target, target_pseudo_label)
                #losses['mlp_target_map_loss'] = self.cls_loss(mlp_dist_target, target_cls_label)
            losses['source_cls_loss'] = self.cls_loss(cls_source, source_cls_label)

        return losses

    def forward_train(self, features_source, features_target, source_label=None, target_label=None, pseudo_label=None):
        self.times_source = int(len(features_source) / len(source_label))
        self.times_target = int(len(features_target) / len(target_label))
        
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

        """
        if self.frozen_map is False:
            self.accumulate_map(mlp_source, mlp_target, source_label, target_label)
        """
        losses = self.loss(features_source,
                           features_target,
                           mlp_source,
                           mlp_target, 
                           cls_source,
                           cls_target,
                           source_label,
                           target_label,
                           pseudo_label,
                           reverse_source,
                           reverse_target)

        return losses

    def distance_test(self, img):
        """Test without augmentation."""
        img_mlp = self.contrastive_projector(img)
        if self.mlp_flag:
            cls_dist = self.mlp_class_map(img_mlp)
        else:
            cls_dist = self.class_map(img)
        pred = F.softmax(cls_dist, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return list(img.detach().cpu().numpy()), list(img_mlp.detach().cpu().numpy()), pred

    def fc_test(self, img):
        img_mlp = self.contrastive_projector(img)
        if self.mlp_flag:
            cls_score = self.fc(img_mlp)
        else:
            cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return list(img.detach().cpu().numpy()), list(img_mlp.detach().cpu().numpy()), pred

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
      
    def sinkhorn_knopp(self, dist):
        Q = torch.exp(dist/self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]
        Q_sum = torch.sum(Q)
        if self.distributed:
            distributed.all_reduce(Q_sum)
        Q /= Q_sum

        #for i in range(self.sinkhorn_iterations):
        for i in range(3):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if self.distributed:
                distributed.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B
    
        Q *= B

        return Q.t()

    """
    def sinkhron_knopp_dual(self, dist):
       u = no.array([14309.,  7365., 16640., 12800.,  9512., 14240., 17360., 12160.,
       10731., 11680., 16000.,  9600.])
    """
