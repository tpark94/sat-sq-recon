'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import argparse
import numpy as np
import random
from pathlib import Path
from scipy.io import savemat

import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import so3_relative_angle
from pytorch3d.loss       import chamfer_distance

import _init_paths

from configs          import cfg, update_config
from dataset.build    import build_dataset
from nets             import Model
from nets.utils       import transform_world2primitive, inside_outside_function_dual
from utils.visualize  import plot_3dmesh, plot_3dpoints, plot_occupancy_labels, imshow
from utils.utils      import (
    set_seeds_cudnn,
    initialize_cuda,
    load_camera_intrinsics
)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training (default: 0)')

    parser.add_argument('--split', default='test', const='test', nargs='?',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate')

    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--modelidx', type=int)
    parser.add_argument('--imageidx', type=int)

    args = parser.parse_args()

    return args


def inference(args, cfg, net, batch):

    with torch.no_grad():
        # In inference mode, mesh only has high-probs primitives, PCL has all
        params, mesh, pcls, trans, rot = net.forward_encoder_generator(batch["image"], train=False)

        # Occupancy function
        occupancy = net.occupancy_function(batch["points_in_mesh"], params) # Fbar
        occupancy_pr = occupancy.sigmoid().cpu() >= 0.5

        # ----- Diff. Render (regardless of config.)
        R       = torch.bmm(rot.transpose(1, 2), net.Rz[0].unsqueeze(0))
        silh    = net.renderer(mesh, R=R, T=trans)
        mask_pr = silh[0, ..., 3].cpu()

        # ----- Find points on surface of assembly
        # Back to primitive-centric for all points
        B, N, M, _ = pcls.shape
        pts_prim = transform_world2primitive(
            pcls.view(B, M * N, -1), params._translation, params._rotation, is_dcm=True
        ) # [B x (N x M) x M x 3]

        # Binary mask of whether the point is on the surface of union of primitives
        F = inside_outside_function_dual(pts_prim, params) # [B x (N x M) x M], < 1 = inside

        # F >= 1 means it's either on surface or outside all surfaces
        F = F.view(B, N, M, M)
        on_surface = (F >= 0.9).all(-1) # [B x N x M]

        # For plots, need separate Pointclouds for different coloring
        pcl_pr = []
        for i in range(cfg.MODEL.NUM_MAX_PRIMITIVES):
            pcl_pr.append(
                Pointclouds(
                    [pcls[0, on_surface[0,:,i], i].cpu()]
                )
            )

        # ----- Separate Meshes
        # For plots, need separate Meshes for different coloring
        mesh_pr = []
        for i in range(cfg.MODEL.NUM_MAX_PRIMITIVES):
            mesh_pr.append(
                Meshes(
                    verts=[pcls[0, :, i]],
                    faces=[net.mesh_converter.faces]
                ).cpu()
            )

        # ----- METRICS
        metrics = {}
        with torch.no_grad():

            # Pose Errors
            metrics['eR'] = so3_relative_angle(rot, batch["rot"])[0].cpu().numpy() # [rad]
            metrics['eT'] = torch.linalg.norm(trans[0] - batch["trans"][0]).cpu().numpy()

            # SPEED Score
            metrics['speed'] = metrics['eR'] + metrics['eT'] / torch.linalg.norm(batch["trans"][0]).cpu().numpy()

            # Chamfer distances
            chamfer_l2, _ = chamfer_distance(
                mesh.verts_padded(), batch["points_on_mesh"], point_reduction='mean', norm=2
            )

            chamfer_l1, _ = chamfer_distance(
                mesh.verts_padded(), batch["points_on_mesh"], point_reduction='mean', norm=1
            )

            metrics['chamfer_l2'] = chamfer_l2.cpu().numpy()
            metrics['chamfer_l1'] = chamfer_l1.cpu().numpy()

            # 3D volumetric IoU
            occ_pred = occupancy[0].any(dim=-1).sigmoid().cpu().numpy() >= 0.5
            occ_true = batch["occ_labels"].cpu().numpy() >= 0.5

            intersection = occ_pred & occ_true
            union        = occ_pred | occ_true
            metrics['iou_3d'] = intersection.sum() / float(union.sum())

            # Number of primitives
            metrics['num_prims'] = sum(params._prob.squeeze().cpu().numpy() > 0.5)

            # 2D IoU
            y_pred = mask_pr.numpy() > 0.5
            y_true = batch['mask'][0].cpu().numpy() > 0.5

            intersection = y_pred & y_true
            union        = y_pred | y_true

            metrics['iou_2d'] = intersection.sum() / float(union.sum())

    return mesh_pr, pcl_pr, mask_pr, occupancy_pr, metrics


def evaluate(cfg):

    args = parse_args()
    update_config(cfg, args)

    print(f"Evaluating for the following split: {args.split.upper()}")

    # Set all seeds & cudNN
    set_seeds_cudnn(cfg, seed=None)

    # GPU device
    device = initialize_cuda(cfg, args.rank)

    # Where to save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ******************************************************************************** #
    # ---------- Build Model
    # ******************************************************************************** #
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    net    = Model(cfg, fov=camera["horizontalFOV"], device=device)

    # Load checkpoint
    load_dict = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location=device)
    net.load_state_dict(load_dict, strict=True)

    # ******************************************************************************** #
    # ---------- Build Dataset
    # ******************************************************************************** #
    dataset = build_dataset(cfg, split=args.split)

    # ******************************************************************************** #
    # ---------- Get Some Results
    # ******************************************************************************** #
    net.eval()

    if args.modelidx is None:
        args.modelidx = random.randrange(dataset.num_models)

    if args.imageidx is None:
        args.imageidx = random.randrange(dataset.num_images_per_model)

    # Get batch
    batch = dataset._get_item(args.modelidx, imgidx=args.imageidx)

    # To CUDA
    batch = {k: v.unsqueeze(0).to(device, non_blocking=True) for k, v in batch.items()}

    # Inference
    mesh_pr, pcl_pr, mask_pr, occupancy_pr, metrics = inference(args, cfg, net, batch)

    # ---------- PLOT
    # Input image
    imshow(batch["image"][0], is_tensor=True, savefn=str(save_dir / "image.jpg"))

    # Mesh
    plot_3dmesh(mesh_pr, markers_for_vertices=False, savefn=str(save_dir / "mesh.jpg"))

    # Metric
    savemat(str(save_dir / "metrics.mat"), metrics)

    print(f"E_T: {metrics['eT']:.2f} [m]    E_R: {np.rad2deg(metrics['eR']):.2f} [deg]    Chamfer-L1 (E-3): {metrics['chamfer_l1'] * 1000:.2f}    Num. prim.: {metrics['num_prims']}")


if __name__=="__main__":
    evaluate(cfg)