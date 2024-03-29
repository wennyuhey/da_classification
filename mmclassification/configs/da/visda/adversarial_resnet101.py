_base_ = [
    '../../_base_/models/da_resnet101.py', 
    '../../_base_/schedules/da_visda.py', '../../_base_/default_runtime.py'
]
model=dict(
    head=dict(
        type='DASupConClsHead',
        num_classes=12,
        in_channels=2048,
        mlp_dim=128,
        #sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.1),
        #dist_loss=dict(type='CosDistLoss', temperature=0.1, loss_weight=0.1),
        #w_loss=dict(type='WDistLoss', loss_weight=1)
        #domain_loss=dict(type='DomainLoss', in_channels=2048, loss_weight=1.0),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        topk=(1)))

#load_from = '/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
load_from = '/lustre/S/wangyu/PretrainedModels/resnet101_gn_ws-3e3c308c_new.pth'
aux = True
