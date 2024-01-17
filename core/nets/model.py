'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import torch
import torch.nn as nn

import logging

from .modules  import *
from .renderer import *
from .losses   import *
from .utils    import transform_world2primitive, inside_outside_function_dual

from pytorch3d.structures import (
    Meshes,
    Pointclouds,
    join_meshes_as_scene,
    join_meshes_as_batch,
    join_pointclouds_as_scene,
    join_pointclouds_as_batch
)

logger = logging.getLogger(__name__)


class Model(nn.Module):

    def __init__(self, cfg, fov, device=torch.device('cpu')):
        super(Model, self).__init__()
        self.device = device
        self.beta   = cfg.LOSS.BETA_OVERLAP
        self.chamfer_gt_num_points = cfg.DATASET.NUM_POINTS_ON_MESH

        self.perform_diff_rendering      = 'reproj' in cfg.LOSS.RECON_TYPE
        self.use_true_pose_during_render = cfg.MODEL.USE_TRUE_POSE_RENDER

        logger.info(f"Creating Model ...")

        # ----- Encoder
        self.encoder = Encoder(cfg.MODEL.LATENT_DIM).to(device)

        # ----- Generator
        self.generator = Generator(cfg.MODEL, fov=fov).to(device)

        # ----- PrimitiveParameters --> Mesh converter
        self.mesh_converter = MeshConverter(
            level=cfg.MODEL.ICOSPHERE_LEVEL,
            dual_sampling=cfg.MODEL.USE_DUAL_SQ,
            device=device
        )

        # ----- Differentiable renderer (silhouette)
        self.renderer = DifferentiableSilhouetteRenderer(
            sigma=cfg.MODEL.RENDER_SIGMA,
            fov=fov,
            image_size=cfg.DATASET.IMAGE_SIZE,
            device=device
        )

        # ----- Loss functions
        self.loss_names   = cfg.LOSS.RECON_TYPE   + cfg.LOSS.POSE_TYPE   + cfg.LOSS.REG_TYPE
        self.loss_weights = cfg.LOSS.RECON_WEIGHT + cfg.LOSS.POSE_WEIGHT + cfg.LOSS.REG_WEIGHT

        self.losses = []
        for name in self.loss_names:
            match name:
                case 'chamfer':
                    self.losses.append(ChamferLoss())
                case 'occupancy':
                    self.losses.append(OccupancyLoss())
                case 'reproj':
                    self.losses.append(nn.MSELoss(reduction='mean'))
                case 'trans':
                    self.losses.append(NormTranslationLoss())
                case 'rot':
                    self.losses.append(RotationLoss())
                case 'overlap':
                    self.losses.append(OverlapReg(cfg.LOSS.BETA_OVERLAP))
                case 'taper':
                    self.losses.append(
                        lambda x : x.flatten(1).square().sum(-1).mean()
                    )

        # ----- Loss helper functions
        self.occupancy_function = OccupancyFunction(
            sharpness=cfg.LOSS.SHARPNESS,
            use_dual_inside_outside=cfg.MODEL.USE_DUAL_SQ,
            device=device
        )

        # ------ Misc.
        # For converting camera frame to pytorch3d convention
        self.Rz = torch.diag(
            torch.tensor([-1, -1, 1], dtype=torch.float32, device=device)
        ).unsqueeze(0)

        logger.info(f"   • Training with diff. rendering: {self.perform_diff_rendering}")
        logger.info(f"   • Rendering with true pose:      {self.use_true_pose_during_render}")
        logger.info(f"   • Using dual superquadrics:      {cfg.MODEL.USE_DUAL_SQ}")
        logger.info(f"   • Number of primitives (M):      {cfg.MODEL.NUM_MAX_PRIMITIVES}")
        logger.info(f"   • Main loss functions:")
        logger.info(f"     - Names:   {cfg.LOSS.RECON_TYPE}")
        logger.info(f"     - Weights: {cfg.LOSS.RECON_WEIGHT}")


    def _convert_params_to_mesh(self, params, train=True, combine_sq_to_one_mesh=True):
        # [B x N x M x 3] Points in world frame
        pts_world = self.mesh_converter.convert(params, is_dcm=True)
        B, _, M   = pts_world.shape[:3]

        # ========== MESH (PYTORCH3D)
        meshes = []
        for b in range(B):
            vs, fs = [], []
            for i in range(M):
                # Add points
                vs.append(pts_world[b, :, i])
                fs.append(self.mesh_converter.faces)

            mesh = Meshes(verts=vs, faces=fs)

            # Either combine multiple SQs to one mesh or return list
            if combine_sq_to_one_mesh:
                meshes.append(
                    join_meshes_as_scene(mesh, include_textures=False)
                )
            else:
                meshes.append(mesh)

        if combine_sq_to_one_mesh:
            meshes = join_meshes_as_batch(meshes, include_textures=False)

        # ========== ON OR BEYOND SURFACES (of ALL primitives)
        with torch.no_grad():
            params_copy    = params.detach()
            pts_world_copy = pts_world.detach()

            # Back to primitive-centric for all points
            B, N, M, _ = pts_world_copy.shape
            pts_prim = transform_world2primitive(
                pts_world_copy.view(B, M * N, -1), params_copy._translation, params_copy._rotation, is_dcm=True
            ) # [B x (N x M) x M x 3]

            # Binary mask of whether the point is on the surface of union of primitives
            F = inside_outside_function_dual(pts_prim, params_copy) # [B x (N x M) x M], < 1 = inside

            # F >= 1 means it's either on surface or outside all surfaces
            F = F.view(B, N, M, M)
            on_surface = (F >= 0.9).all(-1) # [B x N x M]

        # ========== POINT CLOUD (PYTORCH3D)
        # When using pytorch3d's native chamfer_distance function,
        # PCL must be converted to pytorch3d's Pointclouds structure
        pcls = []
        if train:
            for b in range(B):
                pts_batch = []
                for m in range(M):
                    # Add points
                    pts_batch.append(
                        pts_world[b, on_surface[b, :, m], m]
                    )

                if combine_sq_to_one_mesh:
                    pcls.append(
                        join_pointclouds_as_scene(Pointclouds(pts_batch))
                    )
                else:
                    pcls.append(Pointclouds(pts_batch))

            if combine_sq_to_one_mesh:
                pcls = join_pointclouds_as_batch(pcls)

        else:
            pcls = pts_world

        # ========== Return
        return meshes, pcls


    def forward_encoder_generator(self, x, train=True):
        # Encoder
        z_sq, z_pose = self.encoder(x)

        # Generator
        params     = self.generator.forward_sq(z_sq)
        trans, rot = self.generator.forward_pose(z_pose)

        # Params --> SQ mesh
        mesh, pcl = self._convert_params_to_mesh(params, train=train, combine_sq_to_one_mesh=True)

        return params, mesh, pcl, trans, rot


    def forward(self, batch):
        # Image forward
        params, mesh, pcl, trans, rot = self.forward_encoder_generator(batch["image"])

        # Compute occupancy values (without sigmoid)
        if 'occupancy' in self.loss_names or 'overlap' in self.loss_names:
            occupancy = self.occupancy_function(batch["points_in_mesh"], params) # G

        # Diff. rendering
        if 'reproj' in self.loss_names:
            # Convert rotation to Pytorch3D convention
            if self.use_true_pose_during_render:
                rot_render   = torch.bmm(batch["rot"].transpose(1, 2), self.Rz.repeat(rot.shape[0], 1, 1)).detach()
                trans_render = batch["trans"]
            else:
                rot_render   = torch.bmm(rot.transpose(1, 2), self.Rz.repeat(rot.shape[0], 1, 1)).detach()
                trans_render = trans.detach()

            silh = self.renderer(mesh, R=rot_render, T=trans_render)

        # -=-=-=-=-=- Loss Functions -=-=-=-=-=- #
        loss = 0.0
        logs = {}

        for i, loss_fn in enumerate(self.losses):
            match self.loss_names[i]:
                case 'chamfer':
                    l = loss_fn(pcl, batch["points_on_mesh"])

                case 'occupancy':
                    l = loss_fn(occupancy, batch["occ_labels"], batch["occ_weights"])

                case 'reproj':
                    l = loss_fn(silh[..., 3], batch['mask'])

                case 'trans':
                    l = loss_fn(trans, batch['trans'])

                case 'rot':
                    l = loss_fn(rot, batch["rot"])

                case 'overlap':
                    l = loss_fn(occupancy)

                case 'taper':
                    l = loss_fn(params._taper)

            # Add loss
            loss += self.loss_weights[i] * l
            logs[self.loss_names[i]] = l.item()

        return loss, logs