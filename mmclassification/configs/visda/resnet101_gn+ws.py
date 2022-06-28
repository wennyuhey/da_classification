_base_ = [
    '../_base_/models/resnet101.py', '../_base_/datasets/visda.py',
    '../_base_/schedules/visda.py', '../_base_/default_runtime.py'
]
conv_cfg=dict(type='ConvWS')
model = dict(
    type='ImageClassifier',
    backbone=dict(
        conv_cfg = conv_cfg,
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
        frozen_stages=1),
    head=dict(
        num_classes=12
    ))
validate=True
#load_from = '/lustre/S/wangyu/PretrainedModels/resnet101_batch256_imagenet_20200708-753f3608.pth'
load_from = '/lustre/S/wangyu/PretrainedModels/resnet101_gn_ws-3e3c308c_new.pth'
