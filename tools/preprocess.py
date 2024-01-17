'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import numpy as np
import argparse
import trimesh
from tqdm import tqdm

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds

import _init_paths

from configs         import cfg, update_config
from dataset.build   import build_dataset
from utils.visualize import *
from utils.libmesh   import check_mesh_contains


def parse_args():
    parser = argparse.ArgumentParser(description='ArgumentParser for shapeExtractionNet')

    # general
    parser.add_argument('--cfg',
                        help='Experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


if __name__=='__main__':

    args = parse_args()
    update_config(cfg, args)

    # Don't use splits.csv to prepare for all models
    cfg.defrost()
    cfg.DATASET.SPLIT_CSV = None
    cfg.freeze()

    # -=-=-=-=-=- CREATE DATASET STRUCTURE -=-=-=-=-=- #
    dataset = build_dataset(cfg, 'train')

    # -=-=-=-=-=- CREATE PCL FROM MESH SURFACE -=-=-=-=-=- #
    N = 100000
    for idx in tqdm(range(dataset.num_models)):
        d = dataset.datasets[idx]

        # ---------- (1) Sample points on the surface
        points = sample_points_from_meshes(d.mesh, num_samples=N) # [1 x N x 3]
        np.savez(str(d.path_to_surface_points), points=points.numpy())

        plot_3dpoints(
            Pointclouds(points),
            savefn=str(d.path_to_surface_points).replace('npz', 'jpg')
        )

        # ---------- (2) Sample points inside the surface
        mesh_gt = trimesh.load(d.path_to_mesh_file, force='mesh')

        # assert mesh_gt.is_watertight, f'Model {d.tag} is not watertight'

        points = np.random.rand(N, 3) - 0.5
        occupancy = check_mesh_contains(mesh_gt, points)
        np.savez(str(d.path_to_occupancy), points=points, labels=occupancy)

        plot_occupancy_labels(
            points, occupancy,
            savefn=str(d.path_to_occupancy).replace('npz', 'jpg')
        )

        # # ---------- Visualize certain models
        # if d.tag == 'chandra_v09':
        #     with np.load(d.path_to_surface_points) as data:
        #         points = data['points']

        #     plot_3dpoints(Pointclouds(torch.from_numpy(points)))


