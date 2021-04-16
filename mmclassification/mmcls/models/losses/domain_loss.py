import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class DomainLoss(nn.Module):
    def __init__(self, in_channels, loss_weight=1.0):
        super(DomainLoss, self).__init__()
        self.in_channels = in_channels
        self.weight = loss_weight
        self._init_layer()
        self.mse = nn.MSELoss()

    def _init_layer(self):

        self.projector = nn.ModuleList([
            nn.Linear(self.in_channels, self.in_channels),
            nn.Linear(self.in_channels, int(self.in_channels/2))])

        self.domain_classifier = nn.Linear(int(self.in_channels/2), 1)

        self.relu = nn.ReLU(inplace=True)

    def init_weight(self):
        for p in self.projector:
            #normal_init(p, mean=0, std=1./
            nn.init.xavier_uniform_(p.weight)
            nn.init.constant_(p.bias, 0)
        nn.init.xavier_uniform_(self.domain_classifier.weight)
        nn.init.constant_(self.domain_classifier.bias, 0)
 
    def forward(self, features_s, features_t):
        for p in self.projector:
            features_s = self.relu(p(features_s))
            features_t = self.relu(p(features_t))

        domain_s = self.domain_classifier(features_s)
        domain_t = self.domain_classifier(features_t)

        score = torch.cat((domain_s, domain_t))

        gt_s = torch.ones_like(domain_s)
        gt_t = torch.zeros_like(domain_t)

        label = torch.cat((gt_s, gt_t))

        loss = self.mse(score, label)

        losses = {'domain_loss': self.weight * loss}
        return losses
