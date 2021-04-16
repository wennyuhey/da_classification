_base_ = [
    './adversarial_resnet50.py', '../../_base_/datasets/da_normal_a_d.py',
]
#conv_cfg=dict(type='ConvWS')
model = dict(
    backbone = dict(
#        conv_cfg = conv_cfg,
#        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
        frozen_stages = 1))
#load_from = '/lustre/S/wangyu/PretrainedModels/resnet50_gn_ws-15beedd8_new.pth'

