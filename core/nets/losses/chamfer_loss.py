'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch.nn as nn

from pytorch3d.loss import chamfer_distance


class ChamferLoss(nn.Module):
    """ Chamfer loss for superquadratics
    """
    def __init__(self):
        super(ChamferLoss, self).__init__()


    def forward(self, x, y, weights=None):

        loss, _ = chamfer_distance(x, y)

        return loss