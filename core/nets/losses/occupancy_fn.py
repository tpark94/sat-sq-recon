'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch
import torch.nn as nn

from nets.utils import transform_world2primitive, inside_outside_function, inside_outside_function_dual

class OccupancyFunction(nn.Module):
    """ Overlap regularization for superquadratics
    """
    def __init__(self, sharpness=10, use_dual_inside_outside=False, device=torch.device('cpu')):
        super(OccupancyFunction, self).__init__()

        self.sharpness = sharpness
        self.use_dual_inside_outside = use_dual_inside_outside
        self.device = device

    def forward(self, X, params):
        # --- Make sure no points have 0
        X = ((X > 0).float() * 2 - 1) * torch.max(torch.abs(X), X.new_tensor(1.0e-6)).detach()

        # Convert ground-truth points to primitive-centric frames
        # points_tr: [B x Nin x M x 3]
        pts_in_mesh_tr = transform_world2primitive(
            X, params._translation, params._rotation, is_dcm=True
        )

        # Inside/outside values
        # F: [B x N x M]
        # F < 1 : inside
        # F > 1 : outsid
        if self.use_dual_inside_outside:
            F = inside_outside_function_dual(pts_in_mesh_tr, params, untaper=True)
        else:
            F = inside_outside_function(pts_in_mesh_tr, params, untaper=True)

        # G > 0: inside
        # G < 0: outside
        return self.sharpness * (1.0 - F)