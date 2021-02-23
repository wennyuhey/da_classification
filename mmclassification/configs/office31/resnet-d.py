_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/office31-d.py',
    '../_base_/schedules/office.py', '../_base_/default_runtime.py'
]
#model = dict(
#    backbone = dict(
#        frozen_stages = 1))
load_from = '/lustre/S/wangyu/PretrainedModels/resnet50-19c8e357_new.pth'
