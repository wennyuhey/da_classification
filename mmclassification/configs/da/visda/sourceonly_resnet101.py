_base_ = [
    '../../_base_/models/da_resnet101.py', '../../_base_/schedules/cat_visda.py',
    '../../_base_/default_runtime.py', '../../_base_/datasets/categorical_visda.py'
]
model=dict(
    head=dict(
        type='DASupConClsHead',
        num_classes=12,
        in_channels=2048,
        mlp_dim=128,
        threshold=0,
        momentum=0.99,
        sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        frozen_map=False,
        mlp_cls=True,
        topk=(1)))

load_from = '/lustre/S/wangyu/PretrainedModels/resnet101_batch256_imagenet_20200708-753f3608.pth'

aux = True
validation=True
source_only = False
