_base_ = [
    '../../_base_/models/da_resnet101.py', '../../_base_/schedules/da_visda.py',
    '../../_base_/default_runtime.py', '../../_base_/datasets/da_visda.py'
]
model=dict(
    head=dict(
        type='DASupConClsHead',
        num_classes=12,
        in_channels=2048,
        mlp_dim=128,
        sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        combined_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.1),
        #dist_loss=dict(type='CosDistLoss', temperature=0.1, loss_weight=0.1),
        #w_loss=dict(type='WDistLoss', in_channels=128, slice_num=128, loss_weight=1),
        frozen_map=False,
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        topk=(1)))

load_from = '/lustre/S/wangyu/PretrainedModels/resnet101_batch256_imagenet_20200708-753f3608.pth'
aux = True
source_only = False
