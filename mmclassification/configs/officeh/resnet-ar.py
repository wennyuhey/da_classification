_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/home-ar.py',
    '../_base_/schedules/office.py', '../_base_/default_runtime.py'
]
conv_cfg=dict(type='ConvWS')
model = dict(
    head=dict(
        num_classes=65),
    backbone = dict(
        conv_cfg = conv_cfg,
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
        frozen_stages = 1))
load_from = '/lustre/S/wangyu/PretrainedModels/resnet50_gn_ws-15beedd8_new.pth'
validate=True
