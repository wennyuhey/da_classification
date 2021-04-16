# model settings
model = dict(
    type='DASupConClsClassifier',
    backbone = dict(
        type='AuxResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=1),
    neck=dict(type='GlobalAveragePooling'))
