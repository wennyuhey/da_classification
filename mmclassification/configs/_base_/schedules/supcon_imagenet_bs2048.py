# optimizer
optimizer = dict(
    type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25
)
"""
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25,
    step=[300, 600, 900])
"""
runner = dict(type='EpochBasedRunner', max_epochs=1000)
