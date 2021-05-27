_base_ = [
    '../../../_base_/models/da_resnet50.py', '../../../_base_/schedules/cat_office.py',
    '../../../_base_/default_runtime.py', '../../../_base_/datasets/categorical_office_wa.py'
]
model=dict(
    head=dict(
        type='DASupClusterHead',
        num_classes=31,
        in_channels=2048,
        mlp_dim=128,
        cluster=True,
        oracle=False,
        threshold=0,
        momentum=0.9,
        epsilon=0.05,
        bn_projector=False,
        feat_norm=True,
        stable_cost=False,
        cls_map=False,
        balance_trans=False, 
        #sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #combined_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.07, loss_weight=0.1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        pseudo=True,
        frozen_map=True,
        mlp_cls=True,
        topk=(1)))

data = dict(
    train=dict(
        load_mode=dict(target_balance=False,
                       target_shuffle=True,
                       source_balance=True,
                       source_shuffle=False)))


load_from = '/lustre/S/wangyu/PretrainedModels/resnet50_new.pth'
aux = True
validation=True
source_only = False
initialize_pseudo = False
