_base_ = [
    '../../_base_/models/da_resnet50.py', '../../_base_/schedules/classwise_office_source.py',
    '../../_base_/default_runtime.py', '../../_base_/datasets/categorical_office_ad.py'
]
model=dict(
    head=dict(
        type='DASupConClsHead',
        num_classes=31,
        in_channels=2048,
        mlp_dim=128,
        threshold=0,
        sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #combined_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #soft_ce=dict(type='SoftCELoss', loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.3),
        #dist_loss=dict(type='CosDistLoss', temperature=0.1, loss_weight=0.1),
        #w_loss=dict(type='WDistLoss', in_channels=128, slice_num=128, loss_weight=1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        frozen_map=False,
        topk=(1)))

#load_from = 'work_dirs/a_w_resnet50/latest.pth'
load_from = '/lustre/S/wangyu/PretrainedModels/resnet50-19c8e357_new.pth'
#load_from = '/lustre/S/wangyu/env/contrastive/mmclassification/work_dirs/catgorical_office_ad/epoch_51.pth'
aux = True
validation=True
source_only = False
