_base_ = [
    './targetonly_resnet50.py', '../_base_/datasets/da_office31_w_a.py',
]
#conv_cfg=dict(type='ConvWM')
#model=dict(
#    head=dict(
#        type='SupConClsHead',
#        num_classes=31,
#        in_channels=2048,
#        sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.1),
#        con_target_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.1),
#        dist_loss=dict(type='CosDistLoss', temperature=0.1, loss_weight=0.1),
#        cls_loss=dict(type='CrossEntropyLoss', loss_weight=0.1),
#        topk=(1)))

#load_from = '/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
#load_from = '/lustre/S/wangyu/PretrainedModels/resnet50-19c8e357_new.pth'
aux = True
load_from = '/lustre/S/wangyu/checkpoint/classification/da/supcon_sourceonly/office/webcam/latest.pth'
