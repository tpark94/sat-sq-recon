'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch

from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix


# ******************************************************************************** #
# Related to Superquadrics
# ******************************************************************************** #
def transform_primitive2world(xp, translation, rotation, is_dcm=False):
    # Convert point clouds in primitive-centric coordinates
    # to world coordinates (i.e., mesh)
    #
    # xp:          [B x N x M x 3]
    # translation: [B x M x 3]
    # rotation6d:  [B x M x 6]
    #
    # where M = # SQs and N = # points per SQ

    if not is_dcm:
        # rotation = rot_6d_to_matrix(rotation) # [B x M x 3 x 3]
        B, _, M = rotation.shape[:3]
        rotation = quaternion_to_matrix(rotation.view(-1,4)).view(B, M, 3, 3)

    dcm = rotation.unsqueeze(1)    # [B x 1 x M x 3 x 3]
    t   = translation.unsqueeze(1) # [B x 1 x M x 3]

    # (R, t) are world -> primitive, so we need to do
    # xw = R.T * xp + t for all pairs of points & primitives
    xt = xp.unsqueeze(-1)               # [B x N x M x 3 x 1]
    xt = dcm.transpose(3, 4).matmul(xt) # [B x N x M x 3 x 1]

    return xt.squeeze(-1) + t # [B x N x M x 3]


def transform_world2primitive(xw, translation, rotation, is_dcm=False):
    # Convert point clouds in world coordinates (i.e., mesh)
    # to primitive-centric coordinates
    #
    # xw:          [B x N x 3] (world coordinates)
    # translation: [B x M x 3]
    # rotation6d:  [B x M x 6]

    if not is_dcm:
        rotation = rotation_6d_to_matrix(rotation) # [B x M x 3 x 3]
        # B, M = rotation.shape[:2]
        # rotation = quat2dcm(rotation.view(-1,4)).view(B, M, 3, 3)

    dcm = rotation.unsqueeze(1)    # [B x 1 x M x 3 x 3]
    t   = translation.unsqueeze(1) # [B x 1 x M x 3]

    # (R, t) are world -> primitive, so we need to do
    # xp = R(xw - t) for all pairs of points & primitives
    xt = xw.unsqueeze(2)          # [B x N x 1 x 3]

    xt = (xt - t).unsqueeze(-1)   # [B x N x M x 3 x 1]
    xt = dcm.matmul(xt)           # [B x N x M x 3 x 1]

    return xt.squeeze(-1) # [B x N x M x 3]


def untaper_points(x, y, z, a3, k1, k2):
    # pts: [B x N x M x 3]
    # a3:  [B x M]

    s1 = (1 - k1 / a3 * z)
    s2 = (1 - k2 / a3 * z)

    s1 = ((s1 > 0).float() * 2 - 1) * torch.max(torch.abs(s1), s1.new_tensor(1e-6))
    s2 = ((s2 > 0).float() * 2 - 1) * torch.max(torch.abs(s2), s2.new_tensor(1e-6))

    x = x.div(s1)
    y = y.div(s2)

    return x, y


def inside_outside_function(X, params, untaper=True):
    # X : [B x N x M x 3]
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]

    # Decompose
    a1 = params._size[..., 0].unsqueeze(1) # [B x 1 x M]
    a2 = params._size[..., 1].unsqueeze(1)
    a3 = params._size[..., 2].unsqueeze(1)
    e1 = params._shape[..., 0].unsqueeze(1)
    e2 = params._shape[..., 1].unsqueeze(1)

    # Un-taper points
    if untaper and params._taper is not None:
        k1 = params._taper[..., 0].unsqueeze(1)
        k2 = params._taper[..., 1].unsqueeze(1)

        x, y = untaper_points(x, y, z, a3, k1, k2)

    # Prevent INFs
    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

    # Occupancy function
    F = x.div(a1).square().pow(1. / e2)
    F = F + y.div(a2).square().pow(1. / e2)
    F = F.pow(e2 / e1)
    F = F + z.div(a3).square().pow(1. / e1)

    return F.pow(e1)


def inside_outside_function_dual(X, params, untaper=True):
    # X : [B x N x M x 3]
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]

    # Decompose
    a1 = params._size[..., 0].unsqueeze(1) # [B x 1 x M]
    a2 = params._size[..., 1].unsqueeze(1)
    a3 = params._size[..., 2].unsqueeze(1)
    e1 = params._shape[..., 0].unsqueeze(1)
    e2 = params._shape[..., 1].unsqueeze(1)

    # Un-taper points
    if untaper and params._taper is not None:
        k1 = params._taper[..., 0].unsqueeze(1)
        k2 = params._taper[..., 1].unsqueeze(1)

        x, y = untaper_points(x, y, z, a3, k1, k2)

    # Dual inside-outside function
    e1d = 2.0 - e1
    e2d = 2.0 - e2

    two = torch.tensor(2, device=X.device, dtype=torch.float32)

    gamma1 = two.pow(e1d / 2 - 1)
    gamma2 = two.pow(e2d / 2 - 1)

    x = x / (a1 * gamma1 * gamma2)
    y = y / (a2 * gamma1 * gamma2)
    z = z / (a3 * gamma1)

    def _dual_2d(x, y, e):
        a = (x + y) / 2.
        b = (x - y) / 2.

        a = ((a > 0).float() * 2 - 1) * torch.max(torch.abs(a), a.new_tensor(1e-6))
        b = ((b > 0).float() * 2 - 1) * torch.max(torch.abs(b), b.new_tensor(1e-6))

        s = a.square().pow(1. / e) + b.square().pow(1. / e)

        return s.pow(e).sqrt()

    return _dual_2d(_dual_2d(x, y, e2d), z, e1d)


def area_superellipsoid(params):
        # Approximate surface area of superellipsoids by Knud Thomsen
        p = 1.6075
        a = params._size[..., 0]
        b = params._size[..., 1]
        c = params._size[..., 2]

        return 4 * torch.pi * ( ( (a * b).pow(p) + (a * c).pow(p) + (b * c).pow(p) ) / 3 ).pow(1 / p)