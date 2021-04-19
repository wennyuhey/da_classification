# model settings
model = dict(
    type='DASupConClsClassifier',
    backbone = dict(
        type='AuxResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'))
