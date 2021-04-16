_base_ = [
    '../../_base_/models/da_resnet50.py', 
    '../../_base_/schedules/da_office.py', '../../_base_/default_runtime.py'
]
model=dict(
    #wloss=dict(type='WDistLoss', in_channels=128, slice_num=16, loss_weight=1000),
    head=dict(
        type='DASupConClsHead',
        num_classes=31,
        in_channels=2048,
        mlp_dim=128,
        sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=0.1),
        #dist_loss=dict(type='CosDistLoss', temperature=0.1, maxk=1, loss_weight=0.5),
        #w_loss=dict(type='WDistLoss', in_channels=128, slice_num=128, loss_weight=1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=0.1),
        topk=(1)))

#load_from = '/lustre/S/wangyu/PretrainedModels/pretrain_res50x1_new.pth'
#load_from = '/lustre/S/wangyu/PretrainedModels/resnet50-19c8e357_new.pth'
aux = False
