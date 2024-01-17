'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import numpy as np
import torch
import torch.nn as nn

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftSilhouetteShader,
    BlendParams,
)


class DifferentiableSilhouetteRenderer(nn.Module):

    def __init__(
        self,
        sigma=1e-6,
        fov=0.0,
        image_size=256,
        device=torch.device('cpu')
    ):
        """
        For silhouette rendering using Pytorch3D, see

        https://pytorch3d.org/tutorials/fit_textured_mesh
        """
        super(DifferentiableSilhouetteRenderer, self).__init__()

        rasterizer_setting = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=100
        )

        pcamera = FoVPerspectiveCameras(
            fov=fov, degrees=False,
            device=device
        )

        # Silhouette renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=pcamera,
                raster_settings=rasterizer_setting
            ),
            shader=SoftSilhouetteShader(
                BlendParams(sigma=sigma)
            )
        ).to(device)


    def forward(
        self,
        mesh,
        R=torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]),
        T=torch.tensor([[0., 0., 0.]])
    ):

        return self.renderer(mesh, R=R, T=T)
