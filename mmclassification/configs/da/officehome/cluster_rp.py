_base_ = [
    '../../_base_/models/da_resnet50.py', '../../_base_/schedules/cat_office.py',
    '../../_base_/default_runtime.py', '../../_base_/datasets/classwise_officehome_rp.py']
model=dict(
    head=dict(
        type='DASupClusterHead',
        num_classes=65,
        in_channels=2048,
        mlp_dim=128,
        cluster=True,
        oracle=False,
        threshold=0,
        momentum=0.9,
        epsilon=0.05,
        bn_projector=False,
        feat_norm=True,
        stable_cost=False,
        cls_map=True,
        balance_trans=False, 
        #sup_source_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #combined_loss=dict(type='SupConLoss', temperature=0.1, loss_weight=1),
        #con_target_loss=dict(type='SupConLoss', temperature=0.07, loss_weight=0.1),
        cls_loss=dict(type='CrossEntropyLoss', loss_weight=1),
        pseudo=False,
        frozen_map=True,
        mlp_cls=True,
        topk=(1)))

data = dict(
    train=dict(
        load_mode=dict(target_balance=True,
                       target_shuffle=True,
                       source_balance=True,
                       source_shuffle=False)))


#load_from = '/lustre/S/wangyu/PretrainedModels/resnet50-19c8e357_new.pth'
#load_from = '/lustre/S/wangyu/PretrainedModels/resnet50_new.pth'
load_from = 'work_dirs/cluster_rp_sourceonly/epoch_32.pth'
#load_from = '/lustre/S/wangyu/checkpoint/classification/da/classwise/office/sourceonly_mlp_classmap_unnorm/momentum9/latest.pth'
#load_from = 'work_dirs/a_w_resnet50/latest.pth'
#load_from = 'work_dirs/catgorical_office_aw/epoch_73.pth'
aux = True
validation=True
runner = dict(source_only = False)
initialize_pseudo = False
