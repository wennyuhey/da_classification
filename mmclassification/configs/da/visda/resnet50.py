_base_ = [
   './adversarial_resnet101.py', '../../_base_/datasets/da_normal_visda.py',
]
conv_cfg=dict(type='ConvWS')
model = dict(
    backbone = dict(
        conv_cfg = conv_cfg,
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
        frozen_stages = 1))

