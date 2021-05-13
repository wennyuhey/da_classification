_base_ = [
    '../../_base_/models/da_resnet101.py', '../../_base_/schedules/classwise_visda.py',
    '../../_base_/visda_default_runtime.py', '../../_base_/datasets/classwise_visda.py'
]
model=dict(
    head=dict(
        type='DASupClusterHead',
        num_classes=12,
        in_channels=2048,
        mlp_dim=128,
        threshold=0,
        momentum=0.9,
        cluster=True,
        pseudo=True,
        epsilon=0.05,
        oracle=False,
        bn_projector=False,
        feat_norm=True,
        stable_cost=False,
        #sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #combined_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.07, loss_weight=0.1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        frozen_map=True,
        mlp_cls=True,
        topk=(1)))

data = dict(
    train=dict(
        load_mode=dict(target_balance=False,
                       target_shuffle=True,
                       source_balance=True,
                       source_shuffle=False)))


load_from = '/lustre/S/wangyu/PretrainedModels/resnet101_batch256_imagenet_20200708-753f3608.pth'
#resume_from = '/lustre/S/wangyu/checkpoint/classification/da/visda/pseudolabel/singlegpu/norm_eps005_nobn/epoch_5.pth'
#load_from = '/lustre/S/wangyu/checkpoint/classification/da/visda/dist/norm_eps005_nobn/latest.pth'
#resume_from = '/lustre/S/wangyu/checkpoint/classification/da/visda/pseudolabel/singlegpu/norm_eps005_nobn/epoch_11.pth'
aux = True
validation=True
source_only = False
initialize_pseudo = False
