_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/office31-a.py',
    '../_base_/schedules/office.py', '../_base_/default_runtime.py'
]
conv_cfg=dict(type='ConvWM')
model = dict(
    backbone = dict(
#        conv_cfg = conv_cfg,
        #norm_cfg = dict(type='GN', num_groups=32, requires_grad=True), 
        frozen_stages = 1))
load_from = '/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
