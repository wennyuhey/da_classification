import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

def soft_cross_entropy(pred, label):
    log_likelihood = -F.log_softmax(pred, dim=1)
    samples = pred.shape[0]
    loss = torch.sum(torch.mul(log_likelihood, label)) / samples
    return loss

@LOSSES.register_module()
class SoftCELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SoftCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.cls_criterion = soft_cross_entropy
    def forward(self,
                cls_score,
                label,
                **kwargs):
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            **kwargs)
        return loss_cls
