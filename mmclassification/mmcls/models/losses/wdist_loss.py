import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class WDistLoss(nn.Module):
    def __init__(self, in_channels, slice_num, loss_weight=1.0):
        super(WDistLoss, self).__init__()
        self.in_channels = in_channels
        self.weight = loss_weight
        self.slice_num = slice_num
        self._init_layer()

    def _init_layer(self):

#        self.projector = nn.ModuleList([
#            nn.Linear(self.in_channels, self.in_channels),
#            nn.Linear(self.in_channels, self.in_channels)])
        self.classifier = nn.Linear(self.in_channels, self.in_channels)

        #self.slice_p = nn.Linear(self.in_channels, self.slice_num)
        self.slice_p = torch.randn(self.in_channels, self.slice_num).cuda()
        self.slice_p *= torch.rsqrt(torch.sum(torch.mul(self.slice_p, self.slice_p),0,keepdim=True))

        self.relu = nn.ReLU(inplace=True)

    def init_weight(self):
        for p in self.projector:
            #normal_init(p, mean=0, std=1./
            nn.init.xavier_uniform_(p.weight)
            nn.init.constant_(p.bias, 0)

        #self.slice_p.weight.data = torch.randn
 
    def forward(self, features_s, features_t):
#        for p in self.projector:
#            features_s = self.relu(p(features_s))
#            features_t = self.relu(p(features_t))
        features_s = self.classifier(features_s)
        features_t = self.classifier(features_t)
        self.slice_p = torch.randn(self.in_channels, self.slice_num).cuda()
        self.slice_p *= torch.rsqrt(torch.sum(torch.mul(self.slice_p, self.slice_p),0,keepdim=True))

        x_s = torch.matmul(features_s, self.slice_p)
        x_t = torch.matmul(features_t, self.slice_p)
        p1 = torch.topk(x_s, x_s.shape[0], dim=0)[0]
        p2 = torch.topk(x_t, x_t.shape[0], dim=0)[0]
        dist = p1 - p2
        loss = {'wloss': - self.weight * torch.mean(torch.abs(dist))}
        return loss
