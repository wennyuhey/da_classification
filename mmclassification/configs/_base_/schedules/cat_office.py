# optimizer
optimizer_backbone = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_neck = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_fc = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_contrastivep = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_domain = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_map = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='inv', gamma=0.001, power=0.75, by_epoch=False)
runner = dict(type='CatEpochBasedRunner', max_epochs=1000)

