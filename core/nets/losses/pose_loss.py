'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch
import torch.nn as nn

from pytorch3d.transforms import so3_relative_angle


class NormTranslationLoss(nn.Module):
    """ Normalized translation loss for SPEED score
    """
    def __init__(self):
        super(NormTranslationLoss, self).__init__()


    def forward(self, x, y):
        loss = torch.linalg.vector_norm(x - y, ord=2, dim=1)
        loss = loss.div(torch.linalg.vector_norm(y, ord=2, dim=1))

        return loss.mean()


class RotationLoss(nn.Module):
    """ Rotation loss for SPEED score
    """
    def __init__(self):
        super(RotationLoss, self).__init__()


    def forward(self, x, y):
        return so3_relative_angle(x, y).mean()