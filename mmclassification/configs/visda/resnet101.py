_base_ = [
    '../_base_/models/resnet101.py', '../_base_/datasets/visda.py',
    '../_base_/schedules/visda.py', '../_base_/default_runtime.py'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(
        frozen_stages=1),
    head=dict(
        num_classes=12
    ))
validate=True
load_from = '/lustre/S/wangyu/PretrainedModels/resnet101_batch256_imagenet_20200708-753f3608.pth'
