'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import logging
import numpy as np

import torch

from utils.utils import num_trainable_parameters

logger = logging.getLogger(__name__)


def _get_trainable_param(module):
    return filter(lambda p: p.requires_grad, module.parameters())


def get_optimizer(cfg, model):

    logger.info(f'Creating optimizer: {cfg.TRAIN.OPTIMIZER}')
    logger.info(f'   • Initial LR: {cfg.TRAIN.LR}')
    logger.info(f'   • Schedule:   {cfg.TRAIN.SCHEDULER}')

    # Keyword arguments
    kwargs = {'lr': cfg.TRAIN.LR, 'weight_decay': cfg.TRAIN.WD, 'eps': cfg.TRAIN.EPS}
    if cfg.TRAIN.OPTIMIZER in ['Adam', 'AdamW']:
        kwargs['betas'] = (cfg.TRAIN.GAMMA1, cfg.TRAIN.GAMMA2)
    else:
        kwargs['momentum'] = cfg.TRAIN.GAMMA1

    param = _get_trainable_param(model)

    # Create optimizer
    optimizer = getattr(torch.optim, cfg.TRAIN.OPTIMIZER)(param, **kwargs)

    logger.info(f'   • Num. trainable param.: {num_trainable_parameters(model):,d}')

    return optimizer


def get_scaler(cfg):
    scaler = None
    if cfg.AMP and cfg.CUDA:
        scaler = torch.cuda.amp.GradScaler()
        logger.info('Mixed-precision training: ENABLED')

        if cfg.AMP_DTYPE == 'float16':
            amp_dtype = torch.float16
        elif cfg.AMP_DTYPE == 'bfloat16':
            amp_dtype = torch.bfloat16
        else:
            logger.error('AMP_DTYPE must be either float16 or bfloat16')
    else:
        logger.info('Mixed-precision training: DISABLED')
        amp_dtype = torch.float32

    return scaler, amp_dtype


def adjust_learning_rate_epoch(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""

    if cfg.TRAIN.SCHEDULER == 'step':
        # Multi-step decay
        lr_step = cfg.TRAIN.LR_STEP
        if not isinstance(lr_step, (list, tuple)):
            lr_step = [lr_step]

        lr = cfg.TRAIN.LR * (cfg.TRAIN.LR_FACTOR ** sum([epoch >= s for s in lr_step]))

    else:
        NotImplementedError('Only "step" scheduler(s) are implemented for epoch-wise update')

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = cfg.TRAIN.LR
        else:
            param_group['lr'] = lr

    logger.info(f'Current epoch learning rate: {lr:.2e}')


def adjust_learning_rate_step(optimizer, epoch, step, steps_per_epoch, cfg):
    """Decay the learning rate based on schedule"""

    if cfg.TRAIN.SCHEDULER == 'cosine':
        # Cosine annealing
        step_cur = epoch * steps_per_epoch + step
        step_tot = cfg.TRAIN.END_EPOCH * steps_per_epoch

        lr = 0.5 * cfg.TRAIN.LR * (1 + np.cos(step_cur / step_tot * np.pi))

    else:
        NotImplementedError('Only "cosine" scheduler(s) are implemented for step-wise updates')

    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group:
            lr_scale = param_group['lr_scale']
        else:
            lr_scale = 1.0

        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = cfg.TRAIN.LR * lr_scale
        else:
            param_group['lr'] = lr * lr_scale