_base_ = [
    '../_base_/models/da_resnet50.py', '../_base_/datasets/da_office31_a_d.py',
    '../_base_/schedules/da_office.py', '../_base_/default_runtime.py'
]
conv_cfg=dict(type='ConvWM')

#load_from = '/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
load_from = '/lustre/S/wangyu/PretrainedModels/resnet50-19c8e357_new.pth'
