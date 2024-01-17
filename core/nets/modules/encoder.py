'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import torch.nn as nn

import timm

class Encoder(nn.Module):
    ''' Encode image to latent code
    '''
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.encoder = timm.create_model(
            'tf_efficientnet_b0.ns_jft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=[4]
        )

        self.conv = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=4),
            nn.Flatten(1),
            nn.SiLU(inplace=True)
        )

        self.fc1 = nn.Linear(128, latent_dim) # To SQ params
        self.fc2 = nn.Linear(128, latent_dim) # To pose


    def forward(self, x):
        y = self.encoder(x) # [B x 320 x 4 x 4]
        y = self.conv(y[0])

        y_sq   = self.fc1(y)
        y_pose = self.fc2(y)

        return y_sq, y_pose

