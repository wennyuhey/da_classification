import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DADistSamplerSeedHook, build_optimizer, build_runner

from mmcls.core import (DADistEvalHook, DistOptimizerHook, DAEvalHook, OfficeEvalHook,
                        Fp16OptimizerHook, DatasetHook, InitializeHook)
from mmcls.datasets import build_dataloader, build_dataset, build_classwise_dataloader
from mmcls.utils import get_root_logger, convert_splitnorm_model

def classwise_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_classwise_dataloader(
            dataset=ds,
            class_per_iter=cfg.data.class_per_iter,
            batch_size=cfg.data.samples_per_class,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=len(cfg.gpu_ids),
            seed=cfg.seed) for ds in dataset
    ]
    
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = {}
    for name, module in model.module.named_children():
        if 'backbone' in name:
            optimizer.update({name: build_optimizer(module, cfg.optimizer_backbone)})
        elif 'neck' in name:
            continue
        elif 'loss' in name:
            continue
        elif 'head' in name:
            for n, m in module.named_children():
                if n == 'w_loss':
                    optimizer.update({name + '_' + n: build_optimizer(module, cfg.optimizer_w)})
                elif n == 'contrastive_projector':
                    optimizer.update({name + '_' + n: build_optimizer(module, cfg.optimizer_contrastivep)})
                elif n == 'fc':
                    optimizer.update({name + '_' + n: build_optimizer(module, cfg.optimizer_fc)})
                elif 'map' in n:
                    optimizer.update({name + '_' + n: build_optimizer(module, cfg.optimizer_map)})
        else:
            raise ValueError(
                f' "{name}" configuration is not defined in config')

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'DAEpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DADistSamplerSeedHook())

    # register eval hooks
    #if validate:
    val_dataset_s = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataset_t = build_dataset(cfg.data.test, dict(test_mode=True))
    val_dataloader_s = build_dataloader(
        val_dataset_s,
        samples_per_gpu=cfg.data.samples_validate_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    val_dataloader_t = build_dataloader(
        val_dataset_t,
        samples_per_gpu=cfg.data.samples_validate_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    val_dataloader = [val_dataloader_s, val_dataloader_t]

    eval_cfg = cfg.get('evaluation', {})
    #eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    eval_hook = DADistEvalHook if distributed else DAEvalHook
    runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    #initialmap_cfg = cfg.get('initialize', {})
    #initialize_hook = InitializeHook
    #runner.register_hook(initialize_hook(val_dataloader, **initialmap_cfg))

    #cluster_cfg = cfg.get('cluster', {})
    #runner.register_hook(DatasetHook(val_dataloader_t, **cluster_cfg))

    #if cfg.resume_from:
    #    runner.resume(cfg.resume_from)
    if cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if cfg.aux:
        runner.model.module.backbone = convert_splitnorm_model(runner.model.module.backbone)
    #if cfg.load_from:
    #    runner.load_checkpoint(cfg.load_from)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    runner.run(data_loaders, cfg.workflow)
