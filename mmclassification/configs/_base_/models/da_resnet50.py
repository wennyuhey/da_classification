# model settings
model = dict(
    type='SupConClsClassifier',
    backbone = dict(
        type='AuxResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SupConClsHead',
        num_classes=31,
        in_channels=2048,
        sup_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=0.1),
        topk=(1)))
