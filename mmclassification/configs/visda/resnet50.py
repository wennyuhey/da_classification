_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/visda.py',
    '../_base_/schedules/visda.py', '../_base_/default_runtime.py'
]
validate=True
load_from = '/lustre/S/wangyu/PretrainedModels/resnet50-19c8e357_new.pth'
