'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch

from ..utils import transform_primitive2world

from pytorch3d.utils import ico_sphere


def fexp(x, p):
    """ Exponential which preserves sign """
    return torch.sign(x) * torch.pow(torch.abs(x), p)


class MeshConverter(object):
    """ Class to convert base ico-sphere mesh into SQs
    """
    def __init__(self, level=3, dual_sampling=True, device=torch.device('cpu')):

        self.dual_sampling = dual_sampling
        self.device       = device

        base = ico_sphere(level=level, device=device)
        v, f = base.get_mesh_verts_faces(0) # [V x 3], [F x 3]

        self._n_v   = v.shape[0]
        self._faces = f.to(device)

        # Sample angles [N,]
        rxy = torch.linalg.norm(v[..., :2], ord=2, dim=-1)

        self.etas   = torch.atan2(v[..., 2], rxy).to(device)
        self.omegas = torch.atan2(v[..., 1], v[..., 0]).to(device)

        # Remove 0's to prevent nan gradients
        self.etas[self.etas == 0] = 1e-6
        self.omegas[self.omegas == 0] = 1e-6

        self.etas   = self.etas[None, :, None]   # [1, V, 1]
        self.omegas = self.omegas[None, :, None]


    @property
    def faces(self):
        return self._faces


    @property
    def num_vertices(self):
        return self._n_v


    def _convert_from_icosphere(self, params, is_dcm=True):
        # alphas: [B x M x 3]
        alphas, epsilons, trans, rots, prob, taper = params.params

        # Get cartesian coordinates of sampled points
        a1 = alphas[..., 0].unsqueeze(1) # [B x 1 x M]
        a2 = alphas[..., 1].unsqueeze(1)
        a3 = alphas[..., 2].unsqueeze(1)
        e1 = epsilons[..., 0].unsqueeze(1)
        e2 = epsilons[..., 1].unsqueeze(1)

        x = a1 * fexp(torch.cos(self.etas), e1) * fexp(torch.cos(self.omegas), e2)
        y = a2 * fexp(torch.cos(self.etas), e1) * fexp(torch.sin(self.omegas), e2)
        z = a3 * fexp(torch.sin(self.etas), e1)

        # Apply taper
        if taper is not None:
            k1 = taper[..., 0].unsqueeze(1)
            k2 = taper[..., 1].unsqueeze(1)

            x = (1.0 - k1 / a3 * z) * x
            y = (1.0 - k2 / a3 * z) * y

        # Prevent INFs
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

        vertices = torch.stack([x, y, z], dim=-1) # [B x Np x M x 3]

        return transform_primitive2world(vertices, trans, rots, is_dcm=is_dcm)


    def _sample_superellipse_dual(self, a1, a2, e, thetas):
        # a1:     [B x 1 x M]
        # thetas: [1 x N x 1]

        # --- (1) Dual superellipse
        ep    = 2.0 - e
        a1a2  = torch.stack([a1, a2], dim=-1) # [B x 1 x M x 2]
        gamma = torch.tensor(2, device=self.device, dtype=torch.float32).pow(ep / 2 - 1).unsqueeze(-1)

        thetas = thetas + torch.ones_like(thetas) * torch.pi / 4 # NOTE: Important for successful sampling

        ct = fexp(torch.cos(thetas), ep) # [B x N x M]
        st = fexp(torch.sin(thetas), ep)

        v  = torch.stack(
            [
                 ct + st,
                -ct + st
            ], dim=-1
        ) # [B x N x M x 2]

        # --- (3) scale
        v = v * a1a2 * gamma

        return v


    def _convert_using_dual_superellipses(self, params, is_dcm=True):
        # alphas: [B x M x 3]
        alphas, epsilons, trans, rots, prob, taper = params.params

        # Get cartesian coordinates of sampled points
        a1 = alphas[..., 0].unsqueeze(1) # [B x 1 x M]
        a2 = alphas[..., 1].unsqueeze(1)
        a3 = alphas[..., 2].unsqueeze(1)
        e1 = epsilons[..., 0].unsqueeze(1)
        e2 = epsilons[..., 1].unsqueeze(1)

        # Sample from dual superellipses [B x N x M x 2] each
        v1 = self._sample_superellipse_dual(torch.ones_like(a3), a3, e1, self.etas)
        v2 = self._sample_superellipse_dual(a1, a2, e2, self.omegas)

        x = v1[..., 0] * v2[..., 0]
        y = v1[..., 0] * v2[..., 1]
        z = v1[..., 1]

        # Apply taper
        if taper is not None:
            k1 = taper[..., 0].unsqueeze(1)
            k2 = taper[..., 1].unsqueeze(1)

            x = (1.0 - k1 / a3 * z) * x
            y = (1.0 - k2 / a3 * z) * y

        # Prevent INFs
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

        vertices = torch.stack([x, y, z], dim=-1)

        return transform_primitive2world(vertices, trans, rots, is_dcm=is_dcm)


    def convert(self, params, is_dcm=True):

        if self.dual_sampling:
            return self._convert_using_dual_superellipses(params, is_dcm=is_dcm)
        else:
            return self._convert_from_icosphere(params, is_dcm=is_dcm)