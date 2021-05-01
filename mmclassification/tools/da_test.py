import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcls.apis import multi_gpu_test, da_single_gpu_test
from mmcls.core import wrap_fp16_model
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import convert_splitnorm_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metric', type=str, default='accuracy', help='evaluation metric')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data_t.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset_s = build_dataset(cfg.data_s.test)
    data_loader_s = build_dataloader(
        dataset_s,
        samples_per_gpu=cfg.data_t.samples_per_gpu,
        workers_per_gpu=cfg.data_t.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    dataset_t = build_dataset(cfg.data_t.test)
    data_loader_t = build_dataloader(
        dataset_t,
        samples_per_gpu=cfg.data_t.samples_per_gpu,
        workers_per_gpu=cfg.data_t.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)


    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if cfg.aux:
       model.backbone = convert_splitnorm_model(model.backbone) 
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    #if cfg.aux:
    #   model.backbone = convert_splitnorm_model(model.backbone)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        features_s, mlp_features_s, outputs_s = da_single_gpu_test(model, data_loader_s, domain='source', test_mode='fc')
        features_t, mlp_features_t, outputs_t = da_single_gpu_test(model, data_loader_t, domain='target', test_mode='fc')
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.metric != '':
            results_s, _ = dataset_s.evaluate(outputs_s, args.metric, classwise=cfg.evaluation.classwise)
            results_t, _ = dataset_t.evaluate(outputs_t, args.metric, classwise=cfg.evaluation.classwise)
            for topk, acc in results_s.items():
                print(f'\nsource:{topk} accuracy: {acc:.2f}')
            for topk, acc in results_t.items():
                print(f'\ntarget:{topk} accuracy: {acc:.2f}')
        else:
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            if 'CLASSES' in checkpoint['meta']:
                CLASSES = checkpoint['meta']['CLASSES']
            else:
                from mmcls.datasets import ImageNet
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use imagenet by default.')
                CLASSES = ImageNet.CLASSES
            pred_class = [CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if not args.out:
                print('\nthe predicted result for the first element is '
                      f'pred_score = {pred_score[0]:.2f}, '
                      f'pred_label = {pred_label[0]} '
                      f'and pred_class = {pred_class[0]}. '
                      'Specify --out to save all results to files.')
    print(f'\nwriting results to {args.out}')
    mmcv.dump(outputs_s, 'results_s.pkl')
    mmcv.dump(features_s, 'features_s.pkl')
    mmcv.dump(mlp_features_s, 'mlp_features_s.pkl')
    mmcv.dump(dataset_s.get_gt_labels(), 'gt_labels_s.pkl')

    mmcv.dump(outputs_t, 'results_t.pkl')
    mmcv.dump(features_t, 'features_t.pkl')
    mmcv.dump(mlp_features_t, 'mlp_features_t.pkl')
    mmcv.dump(dataset_t.get_gt_labels(), 'gt_labels_t.pkl')

if __name__ == '__main__':
    main()
