import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from Opt import Opt

"""
为了确保生成的x能够服从全体数据的边缘分布
Discriminator for X
输入：来自decoder的x和来自全体数据的x
输出：属于全体数据边缘分布的可能性
"""

class Discriminator_X(nn.Module):
    def __init__(self):
        super(Discriminator_X, self).__init__()
        self.opt = Opt()

        # def conv_block(in_channels, out_channels):
        #     layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=1))]
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     layers.append(nn.Dropout2d(0.25))
        #     return layers

        # self.model = nn.Sequential(
        #     *conv_block(self.opt.img_channels, 128),
        #     *conv_block(128, 256),
        #     *conv_block(256, 512),
        #     *conv_block(512, 1024)
        # )

        def fc_block(in_feat, out_feat):
            layers = [spectral_norm(nn.Linear(in_feat, out_feat))]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *fc_block(self.opt.img_size*self.opt.img_size*self.opt.img_channels, 512),
            *fc_block(512, 512),
            *fc_block(512, 512),
            # *fc_block(512, 512)
        )

        # self.output_layer = spectral_norm(nn.Linear(16*16*1024, 1))
        self.output_layer = spectral_norm(nn.Linear(512, 1))

    def forward(self, img):
        # conv_output = self.model(img)
        # conv_output = conv_output.view(conv_output.shape[0], -1)
        # validity_x = self.output_layer(conv_output)
        img = img.view(img.shape[0], -1)
        validity_x = self.model(img)
        return validity_x