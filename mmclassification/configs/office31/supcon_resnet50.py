_base_ = [
    '../_base_/models/supcon_resnet50.py', '../_base_/datasets/supcon_office31.py',
    '../_base_/schedules/supcon_office.py', '../_base_/default_runtime.py'
]
#load_from='/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
