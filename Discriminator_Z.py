import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from Opt import Opt

"""
为了确保encoder得到的分布趋近于标准正态分布，采用对抗甄别器来最小化wasserstein距离
Discriminator for Z
输入：来自encoder的z和来自标准正态分布的z_normal
输出：属于标准正态分布的可能性
"""

class Discriminator_Z(nn.Module):
    def __init__(self):
        super(Discriminator_Z, self).__init__()
        self.opt = Opt()

        def fc_block(in_feat, out_feat):
            layers = [spectral_norm(nn.Linear(in_feat, out_feat))]
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *fc_block(self.opt.z_size, 512),
            *fc_block(512, 512),
            *fc_block(512, 512),
            *fc_block(512, 512)
        )

        self.output_layer = spectral_norm(nn.Linear(512, 1))

    def forward(self, z):
        fc_output = self.model(z)
        validity_z = self.output_layer(fc_output)
        return validity_z