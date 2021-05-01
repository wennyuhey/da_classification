import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import LINEAR_LAYERS


def norm_linear(input,
               weight,
               bias=None):
    weight = weight / weight.norm(dim=1, keepdim=True)
    return F.linear(input, weight, bias)


@LINEAR_LAYERS.register_module('NormLinear')
class NormLinear(nn.Linear):

    def __init__(self,
                in_features,
                out_features,
                bias=None):
        super(NormLinear, self).__init__(
            in_features,
            out_features,
            bias=bias)

    def forward(self, x):
        return norm_linear(x, self.weight, self.bias)

