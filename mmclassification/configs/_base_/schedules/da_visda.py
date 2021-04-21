# optimizer
optimizer_backbone = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_neck = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
#optimizer_head = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
optimizer_fc = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_w = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_domain = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_contrastivep = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='inv', gamma=0.0005, power=2.25, by_epoch=False)
runner = dict(type='DAEpochBasedRunner', max_epochs=100, source_only=False)

