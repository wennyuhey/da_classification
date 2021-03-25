_base_ = [
    '../_base_/models/supcon_resnet50.py', '../_base_/datasets/supcon_imagenet_bs64.py',
    '../_base_/schedules/supcon_imagenet_bs2048.py', '../_base_/supcon_runtime.py'
]
model=dict(
    head=dict(
        type='SupConHead',
        num_classes=1000,
        in_channels=2048,
        supcon_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1.),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1.),
        topk=(1)))

validate=False

