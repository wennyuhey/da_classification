# dataset settings
dataset_type = 'ClasswiseVisDA'
dataset_type_val = 'VisDA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, prob=0.8),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label', 'pseudo_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    workers_per_gpu=2,
    class_per_iter=12,
    samples_per_class=8,
    samples_validate_per_gpu=500,
    train=dict(
        type=dataset_type,
        data_prefix='data/visda/',
        source_prefix='train',
        target_prefix='validation',
        times=2,
        load_mode=dict(target_balance=False,
                       target_shuffle=True,
                       source_balance=True,
                       source_shuffle=False),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type_val,
        data_prefix='data/visda/train',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type_val,
        data_prefix='data/visda/validation',
        pipeline=test_pipeline))
evaluation = dict(by_epoch=True, classwise=12, test_mode='distance', interval=1, metric='accuracy', metric_options=dict(topk=(1)))
initialize = dict(by_epoch=True, interval=1)
