'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch.nn as nn


class OverlapReg(nn.Module):
    """ Regularization for overlapping primitives
    """
    def __init__(self, overlap_beta=1.95):
        super(OverlapReg, self).__init__()
        self.beta = overlap_beta

    def forward(self, G):
        loss = (G.sigmoid().sum(-1) - self.beta).relu().mean()

        return loss