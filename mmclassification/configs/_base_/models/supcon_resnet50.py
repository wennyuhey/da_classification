# model settings
model = dict(
    type='SupConClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SupConHead',
        loss=dict(type='SupConLoss', temperature=0.1)
    ))
