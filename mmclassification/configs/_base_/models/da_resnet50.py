# model settings
model = dict(
    type='SupConClsClassifier',
    backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=1),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SupConClsHead',
        num_classes=31,
        in_channels=2048,
        sup_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1.0),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
