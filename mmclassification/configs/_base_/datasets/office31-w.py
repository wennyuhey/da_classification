# dataset settings
dataset_type = 'PartialOffice'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
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
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/office31/webcam/images',
        pipeline=train_pipeline),
    val_a=dict(
        type=dataset_type,
        data_prefix='data/office31/amazon/images',
        pipeline=test_pipeline),
    val_w=dict(
        type=dataset_type,
        data_prefix='data/office31/webcam/images',
        pipeline=test_pipeline),
    val_d=dict(
        type=dataset_type,
        data_prefix='data/office31/dslr/images',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/office31/webcam/images',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=(1)))
