_base_ = [
    '../_base_/models/supcon_resnet50.py', '../_base_/datasets/supcon_office31_w.py',
    '../_base_/schedules/supcon_office.py', '../_base_/default_runtime.py'
]
model=dict(
    head=dict(
        type='SupConClsHead',
        num_classes=31,
        in_channels=2048,
        mlp_dim=128,
        supcon_loss=dict(type='SupConLoss', loss_weight=1.0),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,))
)
validate=True
load_from='/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
