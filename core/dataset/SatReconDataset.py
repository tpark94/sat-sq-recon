'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import numpy as np
import random
import cv2
from pathlib import Path
import logging
import pandas as pd

import torch
from pytorch3d.transforms import quaternion_to_matrix

from .base import SPE3R

logger = logging.getLogger(__name__)

class SatReconDataset(torch.utils.data.Dataset):
    ''' torch Dataset class to be loaded
    '''
    def __init__(self, cfg, split='train', transforms=None):
        super(SatReconDataset, self).__init__()

        assert split in ['train', 'validation', 'test']

        self.root_dir   = Path(cfg.DATASET.ROOT) / cfg.DATASET.DATANAME
        self.split      = split
        self.is_train   = split == 'train'
        self.transforms = transforms

        # Misc
        self.rgb = True
        self.num_points_on_mesh = cfg.DATASET.NUM_POINTS_ON_MESH
        self.num_points_in_mesh = cfg.DATASET.NUM_POINTS_IN_MESH

        # NOTE: For ShapeNet, cfg.DATASET.DATANAME = "ShapeNetV2Core/<model>"
        dataset_name = cfg.DATASET.DATANAME

        # When there are splits, tags should only contains those in CSV file
        if cfg.DATASET.SPLIT_CSV is not None:
            csv = pd.read_csv(str(self.root_dir / cfg.DATASET.SPLIT_CSV), header=None)

            temp = 'train' if split == 'train' or split == 'validation' else 'test'
            tags = [
                csv.iloc[idx, 0] for idx in range(len(csv)) if csv.iloc[idx, 1] == temp
            ]
        else:
            tags = None

        # Dataset object
        self.datasets = SPE3R(cfg, tags=tags)

        # NOTE: SPE3R has 1,000 images total, set 200 apart for validation
        # - Train:      0001 ~ 0400, 0501 ~ 0900
        # - Validation: 0401 ~ 0500, 0901 ~ 1000
        if split == 'train':
            self.image_indices = list(range(400)) + list(range(500, 900))
        elif split == 'validation':
            self.image_indices = list(range(400, 500)) + list(range(900, 1000))
        else:
            self.image_indices = list(range(1000))

        logger.info(f"Dataset: {dataset_name} ({split})")
        logger.info(f"   • Num. models:         {self.num_models}")
        logger.info(f"   • Num. images / model: {self.num_images_per_model}")
        logger.info(f"   • Num. GT pts ON mesh: {self.num_points_on_mesh}")
        logger.info(f"   • Num. GT pts IN mesh: {self.num_points_in_mesh}")


    @property
    def num_models(self):
        return len(self.datasets)


    @property
    def num_images_per_model(self):
        return len(self.image_indices)


    def __len__(self):
       # Number of models * number of images per model
       return len(self.datasets) * self.num_images_per_model


    def __getitem__(self, idx):
        modelidx = int(idx / self.num_images_per_model)
        imageidx = idx % self.num_images_per_model

        batch = self._get_item(modelidx, imgidx=imageidx)

        return batch


    def _get_item(self, modelidx, imgidx=None):
        # Grab idx'th MODEL
        dataset = self.datasets[modelidx]

        # ---------- Get image & mask
        if imgidx is None:
            imgidx = random.randrange(self.num_images_per_model)

        # Get correct image index
        imgidx = self.image_indices[imgidx]

        # NOTE: Just to make sure we got image indexing right
        if self.split == 'train':
            assert (imgidx >= 0 and imgidx < 400) or (imgidx >= 500 and imgidx < 900), \
                f"Got imgidx = {imgidx} for {self.split} split"
        elif self.split == 'validation':
            assert (imgidx >= 400 and imgidx < 500) or (imgidx >= 900 and imgidx < 1000), \
                f"Got imgidx = {imgidx} for {self.split} split"

        # Load image & mask
        image = self._load_image(dataset.random_imagepath(ri=imgidx))
        mask  = self._load_mask(dataset.random_maskpath(ri=imgidx))

        # Blur out mask
        mask = cv2.blur(mask, (5, 5))

        # Apply transform
        if self.transforms:
            transform_kwargs = {
                'image': image,
                'mask':  mask
            }

            transformed = self.transforms(**transform_kwargs)
            image = transformed['image']
            mask  = transformed['mask']
        else:
            raise AssertionError('We need transformations for pre-processing')

        # Mask to CHW
        thresh = 0.2
        mask[mask >  thresh] = 1
        mask[mask <= thresh] = 0

        # --------- Get pose
        trans, quat = dataset.random_pose(ri=imgidx)

        trans = torch.tensor(trans, dtype=torch.float32)
        quat  = torch.tensor(quat,  dtype=torch.float32)
        rot   = quaternion_to_matrix(quat)

        # ---------- Get mesh surface points
        surface = np.load(
            dataset.path_to_surface_points,
            allow_pickle=True
        )
        pidx = random.sample(list(range(100000)), self.num_points_on_mesh)
        points_on_mesh = torch.tensor(surface["points"][0, pidx], dtype=torch.float32)

        # ---------- Get occupancy labels
        occupancy = np.load(
            dataset.path_to_occupancy,
            allow_pickle=True
        ) # [N,]

        points     = occupancy["points"]
        occ_labels = occupancy["labels"]

        n_positive = occ_labels.sum()

        if n_positive < self.num_points_in_mesh / 2:
            # Not enough positive points -- use all
            idx_positive = np.where(occ_labels == 1)[0]
            idx_negative = np.random.choice(np.where(occ_labels == 0)[0], self.num_points_in_mesh - n_positive)
        else:
            # Enough positive points -- sample
            idx_positive = np.random.choice(np.where(occ_labels == 1)[0], int(self.num_points_in_mesh / 2))
            idx_negative = np.random.choice(np.where(occ_labels == 0)[0], int(self.num_points_in_mesh / 2))

        pidx = np.concatenate([idx_positive, idx_negative])
        assert len(pidx) == self.num_points_in_mesh

        points_in_mesh = torch.tensor(points[pidx],     dtype=torch.float32)
        occ_labels     = torch.tensor(occ_labels[pidx], dtype=torch.float32)
        occ_weights    = torch.ones_like(occ_labels)

        batch = {
            'image':  image,
            'mask':   mask.float(),
            'points_on_mesh': points_on_mesh,
            'points_in_mesh': points_in_mesh,
            'occ_labels':  occ_labels,
            'occ_weights': occ_weights,
            'trans':  trans,
            'rot':    rot
        }

        return batch


    def _load_image(self, fn):
        """ Read image of given index from a folder, if specified """
        data = cv2.imread(fn, cv2.IMREAD_COLOR)

        if self.rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            # Force grayscale
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

        return data


    def _load_mask(self, fn):
        """ Read mask image """
        data = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

        # Clean up any intermediate values
        data[data >  128] = 255
        data[data <= 128] = 0

        return data[:,:,None]
        # return data