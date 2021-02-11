_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/office31-a.py',
    '../_base_/schedules/office.py', '../_base_/default_runtime.py'
]
load_from='/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
