import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DADistSamplerSeedHook, build_optimizer, build_runner

from mmcls.core import (DADistEvalHook, DistOptimizerHook, DAEvalHook, OfficeEvalHook,
                        Fp16OptimizerHook)
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger, convert_splitnorm_model


def da_set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def da_train_model(model,
                dataset_s,
                dataset_t,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset_s = dataset_s if isinstance(dataset_s, (list, tuple)) else [dataset_s]
    dataset_t = dataset_t if isinstance(dataset_t, (list, tuple)) else [dataset_t]

    data_loaders_s = [
        build_dataloader(
            ds,
            cfg.data_s.samples_per_gpu,
            cfg.data_s.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
            drop_last=True,
            seed=cfg.seed) for ds in dataset_s
    ]

    if cfg.source_only:
       data_loaders_t = None
    else:
        data_loaders_t = [
            build_dataloader(
                ds,
                cfg.data_t.samples_per_gpu,
                cfg.data_t.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                round_up=True,
                drop_last=True,
                seed=cfg.seed) for ds in dataset_t
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
            #optimizer.update({name: build_optimizer(module, cfg.optimizer_neck)})
            continue
        elif 'loss' in name:
            continue
        elif 'head' in name:
            for n, m in module.named_children():
                if n == 'w_loss':
                    optimizer.update({name + '_' + n: build_optimizer(m, cfg.optimizer_w)})
                elif n == 'contrastive_projector':
                    optimizer.update({name + '_' + n: build_optimizer(m, cfg.optimizer_contrastivep)})
                elif n == 'fc':
                    optimizer.update({name + '_' + n: build_optimizer(m, cfg.optimizer_fc)})
                elif n == 'domain_loss':
                    optimizer.update({name + '_' + n: build_optimizer(m, cfg.optimizer_domain)})
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
    if validate:
        val_dataset_s = build_dataset(cfg.data_s.val, dict(test_mode=True))
        val_dataset_t = build_dataset(cfg.data_t.val, dict(test_mode=True))
        val_dataloader = []
        val_dataloader.append(build_dataloader(
            val_dataset_s,
            samples_per_gpu=cfg.data_s.samples_validate_per_gpu,
            workers_per_gpu=cfg.data_s.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            round_up=True))
        val_dataloader.append(build_dataloader(
            val_dataset_t,
            samples_per_gpu=cfg.data_t.samples_validate_per_gpu,
            workers_per_gpu=cfg.data_t.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            round_up=True))

        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DADistEvalHook if distributed else DAEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

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
    runner.run(data_loaders_s, data_loaders_t, cfg.workflow)
