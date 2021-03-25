_base_ = [
    './targetonly_resnet50.py', '../_base_/datasets/da_office31_a_w.py',
]
load_from = '/lustre/S/wangyu/env/contrastive/mmclassification/work_dirs/a_resnet50/epoch_20.pth'
