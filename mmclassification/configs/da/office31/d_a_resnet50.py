_base_ = [
    './targetonly_resnet50.py', '../_base_/datasets/da_office31_d_a.py',
]
load_from = '/lustre/S/wangyu/checkpoint/classification/da/supcon_sourceonly/office/dslr/epoch_20.pth'
