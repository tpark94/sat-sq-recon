'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import json
import numpy as np
from pathlib import Path
import logging

from pytorch3d.io import load_objs_as_meshes

logger = logging.getLogger(__name__)


'''
Dataset structure is as follows:

Dataset
    - Model #1 tag
        - images
        - masks
        - model
    - Model #2 tag
    ...

e.g., Dataset = SPE3R,                   tag = Aquarius
      Dataset = ShapeNetCoreV2/04401088, tag = 1a0fab14a11b39d1a5295d0078b5d60
'''

class BaseModelDataset(object):
    ''' Base python class to hold all info (e.g., images, meshes) for EACH model
    '''
    def __init__(
        self, tag, path_to_model_dir, image_dir="images", mask_dir="masks", mesh=None, image_size=(256, 256)
    ):
        self.tag        = tag
        self.image_size = image_size

        # Mesh
        self.path_to_mesh_file = Path(path_to_model_dir) / "models" / "model_normalized.obj"
        self._mesh_gt = mesh

        # Image & pose paths
        self.path_to_image_dir = Path(path_to_model_dir) / image_dir
        self.path_to_mask_dir  = Path(path_to_model_dir) / mask_dir
        path_to_pose_json      = Path(path_to_model_dir) / "labels.json"

        # Other paths
        self.path_to_surface_points = Path(path_to_model_dir) / "surface_points.npz"
        self.path_to_occupancy      = Path(path_to_model_dir) / "occupancy_points.npz"

        # Read .json
        if path_to_pose_json.exists():
            with open(str(path_to_pose_json), 'r') as f:
                self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    @property
    def mesh(self):
        if self._mesh_gt is None:
            self._mesh_gt = load_objs_as_meshes([self.path_to_mesh_file], load_textures=False)
        return self._mesh_gt

    def random_imagepath(self, ri=None):
        if ri is None:
            ri = np.random.choice(len(self))
        return str(self.path_to_image_dir / (self.labels[ri]["filename"] + ".jpg"))

    def random_maskpath(self, ri=None):
        if ri is None:
            ri = np.random.choice(self.num_masks)
        return str(self.path_to_mask_dir / (self.labels[ri]["filename"] + ".png"))

    def random_pose(self, ri=None):
        if ri is None:
            ri = np.random.choice(self.num_masks)
        return self.labels[ri]["r_Vo2To_vbs_true"], self.labels[ri]["q_vbs2tango_true"]


class SPE3R(object):
    ''' Creates BaseModelDataset for each model
    '''
    def __init__(self, cfg, tags=None):

        base_dir = Path(cfg.DATASET.ROOT) / cfg.DATASET.DATANAME

        # List of PosixPath's (absolute path) for tags
        all_paths = sorted(
            x for x in Path(base_dir).iterdir() if x.is_dir()
        )

        # List of tags
        tags_in_dir = [p.name for p in all_paths]

        if tags:
            # Custom list of tags (e.g., those in train & val)
            self._tags  = [t for t in tags if t in tags_in_dir]
            self._paths = [p for p in all_paths if p.name in self._tags]
        else:
            self._tags  = tags_in_dir
            self._paths = all_paths

        # Misc.
        self.image_dir  = cfg.DATASET.IMAGE_DIR
        self.mask_dir   = cfg.DATASET.MASK_DIR
        self.image_size = cfg.DATASET.IMAGE_SIZE

        # Pre-load
        self.datasets = []
        for i in range(len(self)):
            self.datasets.append(
                BaseModelDataset(
                    self._tags[i],
                    self._paths[i],
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                    mesh=None,
                    image_size=self.image_size
                )
            )

    def _get_base_model_of_tag(self, i, mesh=None):
        if not self.datasets:
            return self.datasets[i]
        else:
            return BaseModelDataset(
                self._tags[i],
                self._paths[i],
                image_dir=self.image_dir,
                mask_dir=self.mask_dir,
                mesh=mesh,
                image_size=self.image_size
            )

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, i):
        return self._get_base_model_of_tag(i)