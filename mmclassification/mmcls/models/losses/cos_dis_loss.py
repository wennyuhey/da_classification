import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class CosDistLoss(nn.Module):
    def __init__(self, temperature=0.07, maxk=1, loss_weight=1.0):
        super(CosDistLoss, self).__init__()
        self.temperature = temperature
        self.weight = loss_weight
        self.maxk = maxk

    def forward(self, features, class_map):
        """
        log_class_prob = []
        class_dot_dist = torch.div(torch.matmul(features, class_map.T)/class_map.norm(dim=1), self.temperature)
        #dist, idx = class_dot_dist.topk(self.maxk, dim=1)
        dist_max, dist_max_idx = torch.max(class_dot_dist, dim=1, keepdim=True)
        #class_dot_dist = class_dot_dist - dist_max
        exp_dist = torch.exp(- class_dot_dist)
        #exp_dist_max = torch.exp(dist_max)
        #log_class_prob = torch.log(exp_dist_max.sum(1, keepdim = True)) - torch.log(exp_dist.sum(1, keepdim=True)) +\ 
        log_class_prob = - dist_max - torch.log(exp_dist.sum(1, keepdim=True))
        loss = - self.weight * log_class_prob.mean()

        return loss
        """
        log_class_prob = []
        class_dot_dist = torch.div(torch.matmul(features, class_map.T), self.temperature)
        #dist, idx = class_dot_dist.topk(self.maxk, dim=1)
        dist_max, dist_max_idx = torch.max(class_dot_dist, dim=1, keepdim=True)
        class_dot_dist = class_dot_dist - dist_max
        exp_dist = torch.exp(class_dot_dist)
        #exp_dist_max = torch.exp(dist)
        #log_class_prob = torch.log(exp_dist_max.sum(1, keepdim = True)) - torch.log(exp_dist.sum(1, keepdim=True)) +\ 
        log_class_prob = - torch.log(exp_dist.sum(1, keepdim=True))
        loss = - self.weight * log_class_prob.mean()

        return loss
