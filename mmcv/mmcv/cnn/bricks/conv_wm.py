import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CONV_LAYERS


def conv_wm_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    #c_in = weight.size(0)
    #weight_flat = weight.view(c_in, -1)
    #weight = weight / torch.norm(weight_flat)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


@CONV_LAYERS.register_module('ConvWM')
class ConvWM2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(ConvWM2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_wm_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)

