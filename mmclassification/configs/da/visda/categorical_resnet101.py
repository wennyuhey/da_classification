_base_ = [
    '../../_base_/models/da_resnet101.py', '../../_base_/schedules/cat_visda.py',
    '../../_base_/default_runtime.py', '../../_base_/datasets/categorical_visda.py'
]
model=dict(
    head=dict(
        #type='DASupConClsHead',
        type='DASupClusterHead',
        num_classes=12,
        in_channels=2048,
        mlp_dim=128,
        threshold=0.8,
        momentum=0.9,
        #sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #combined_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #soft_ce=dict(type='SoftCELoss', loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.3),
        #dist_loss=dict(type='CosDistLoss', temperature=0.1, loss_weight=0.1),
        #w_loss=dict(type='WDistLoss', in_channels=128, slice_num=128, loss_weight=1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        frozen_map=False,
        mlp_cls=True,
        topk=(1)))

data = dict(
    train=dict(
        load_mode=dict(target_balance=True,
                       target_shuffle=True,
                       source_balance=True,
                       source_shuffle=False)))

load_from = '/lustre/S/wangyu/checkpoint/classification/da/classwise/visda/sourceonly_fc_classmap_unnorm/momentum99/epoch_1.pth'
#load_from = '/lustre/S/wangyu/checkpoint/classification/da/classwise/cls+supcon/map/epoch_2.pth'
aux = True
validation=True
source_only = False
