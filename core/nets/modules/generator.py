'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import rotation_6d_to_matrix

from .layers import *


class Generator(nn.Module):

    def __init__(
        self, cfg, fov=0.1
    ):
        super(Generator, self).__init__()

        self.n_parts     = cfg.NUM_MAX_PRIMITIVES
        self.apply_taper = cfg.APPLY_TAPER
        self.use_dual_sq = cfg.USE_DUAL_SQ

        # 2 [shape] + 3 [size] + 3 [trans] + 6 [rot] + 1 [prob]
        self.sq_len = 15
        if self.apply_taper:
            self.sq_len += 2

        def block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, cfg.HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.HIDDEN_DIM, out_dim)
            )

        # Blocks for each SQ
        self.blocks_sq = nn.ParameterList(
            [
                block(cfg.LATENT_DIM, self.sq_len) for _ in range(self.n_parts)
            ]
        )

        # Block for pose
        self.block_pose = block(cfg.LATENT_DIM, 9)

        # NOTE: Ad hoc method to restrict max & min distance to target based on dataset
        # - The target occupies 75 ~ 100% of FoV
        # - The target fits within a unit cube
        radius = 0.5
        self.min_dist_z = radius / np.tan(fov / 2.0)
        self.max_dist_z = radius / 0.75 / np.tan(fov / 2.0)

        # Max distance along image plane
        self.max_dist_xy = np.tan(fov / 2.0) * self.min_dist_z


    def forward_sq(self, x):
        B      = x.shape[0]
        device = x.device

        params_all = []
        for i in range(self.n_parts):
            params = self.blocks_sq[i](x)

            # Decode parameters
            shape = F.sigmoid(params[:,:2])                # shape:       [0, 1]
            size  = F.sigmoid(params[:,2:5]) * 0.49 + 0.01 # size:        [0.01, 0.5]
            trans = F.tanh(params[:,5:8]) * 0.5            # translation: [-0.5, 0.5]
            rot   = rotation_6d_to_matrix(params[:,8:14])

            if not self.use_dual_sq:
                shape = shape * 0.8 + 0.2 # Primal param.: [0.2, 1.0]

            # Probability -- all ones for now
            prob = torch.ones_like(params[:,14]).unsqueeze(-1)

            taper = None
            if self.apply_taper:
                taper = F.tanh(params[:,15:17]) * 0.9 # linear taper: [-0.9, 0.9]

            params = PrimitiveParameters(
                shape, size, trans, rot, prob, taper=taper
            )

            # Save
            params_all.append(params)

        # Stack parameters
        params = PrimitiveParameters(
            torch.zeros(B, self.n_parts, 2, device=device),
            torch.zeros(B, self.n_parts, 3, device=device),
            torch.zeros(B, self.n_parts, 3, device=device),
            torch.zeros(B, self.n_parts, 3, 3, device=device),
            torch.zeros(B, self.n_parts, 1, device=device),
            torch.zeros(B, self.n_parts, 2, device=device) if self.apply_taper else None
        )

        for i, p in enumerate(params_all):
            params._shape[:,i] = p._shape
            params._size[:,i]  = p._size
            params._translation[:,i] = p._translation
            params._rotation[:,i]    = p._rotation
            params._prob[:,i] = p._prob

            if self.apply_taper:
                params._taper[:,i] = p._taper

        return params


    def forward_pose(self, x):
        pose = self.block_pose(x)

        trans = pose[..., :3]
        rot   = pose[..., 3:]

        # XY-direction limited
        # NOTE: Even though dataset has 0 translation in XY, still predict
        # xy = F.tanh(trans[..., :2]) * self.max_dist_xy
        xy = torch.zeros_like(trans[..., :2])

        # Z-direction limited
        z = F.sigmoid(trans[..., 2]) * (self.max_dist_z - self.min_dist_z) + self.min_dist_z

        return torch.cat([xy, z.unsqueeze(-1)], dim=-1), rotation_6d_to_matrix(rot)


    def forward(self, x):
        params     = self.forward_sq(x)
        trans, rot = self.forward_pose(x)

        return params, trans, rot
