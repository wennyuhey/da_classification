import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class CosDistLoss(nn.Module):
    def __init__(self, temperature=0.07, loss_weight=1.0):
        super(CosDistLoss, self).__init__()
        self.temperature = temperature
        self.weight = loss_weight

    def forward(self, features, class_map):

        class_dot_dist = torch.div(torch.matmul(features, class_map.T), self.temperature)
        dist_max, dist_max_idx = torch.max(class_dot_dist, dim=1, keepdim=True)
        class_dot_dist = class_dot_dist - dist_max
        exp_dist = torch.exp(class_dot_dist)
        log_class_prob = - torch.log(exp_dist.sum(1, keepdim=True))
        loss = - self.weight * log_class_prob.mean()

        return loss
