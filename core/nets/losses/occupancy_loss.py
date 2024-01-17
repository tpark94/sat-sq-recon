'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch.nn as nn

class OccupancyLoss(nn.Module):
    """ Occupancy loss for superquadratics
    """
    def __init__(self):
        super(OccupancyLoss, self).__init__()


    def forward(self, G, labels, weights):
        # G: [B x N x M] (Occupancy -- w/o sigmoid)
        # G > 0: inside
        # G < 0: outside
        #
        # probs:   existence probabilities [B x M x 1]
        # labels:  inside/outside labels   [B x N]
        # weights: weights for each labels [B x N]

        # Find closest primitive for each point
        Gunion = G.max(-1)[0] # [B x Nin]

        # BCE loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            Gunion, labels, weight=weights, reduction='mean'
        )

        return loss