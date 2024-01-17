'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import torch
import numpy as np
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .SatReconDataset import SatReconDataset


def _seed_worker(worker_id):
    """ Set seeds for dataloader workers. For more information, see below
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(cfg, split='train'):

    # TODO: create dedicated image transform script later
    transforms = [
        A.Resize(cfg.DATASET.IMAGE_SIZE[1], cfg.DATASET.IMAGE_SIZE[0]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    transforms = A.Compose(transforms)

    dataset = SatReconDataset(cfg, split, transforms=transforms)

    return dataset


def get_dataloader(cfg, split='train', distributed=False):

    # TODO: Temporary
    assert not distributed

    if split=='train':
        images_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle        = cfg.TRAIN.SHUFFLE
        num_workers    = min(cfg.TRAIN.BATCH_SIZE_PER_GPU, cfg.TRAIN.WORKERS)
    elif split == 'validation':
        images_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle        = False
        num_workers    = min(cfg.TEST.BATCH_SIZE_PER_GPU, cfg.TRAIN.WORKERS)
    else:
        images_per_gpu = 1
        shuffle        = False
        num_workers    = 0

    dataset = build_dataset(cfg, split)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker
    )

    return data_loader