# optimizer
optimizer_backbone = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_neck = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_head = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
#optimizer_fc = dict(
#    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25
)
runner = dict(type='EpochBasedRunner', max_epochs=1000)
