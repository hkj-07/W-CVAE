import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from Opt import Opt

"""
解码器
输入：Z和标签
输出：对应类别的x
"""

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.opt = Opt()

        self.embed_z = spectral_norm(nn.Linear(self.opt.z_size+self.opt.n_classes, 7*7*1024))

        def conv_block(in_channels, out_channels, bn=True, up=True):
            layers = []
            if up:
                # 每次上采样，图像边长翻倍 7 14 28 只能翻倍两次
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *conv_block(1024, 512),
            *conv_block(512, 256),
            *conv_block(256, self.opt.img_channels, up=False)
        )

    def forward(self, z, label):
        # 将z和label连起来
        catted_z = torch.cat([z, label], dim=1)
        # 映射到512*7*7的图片
        conv_input = self.embed_z(catted_z)
        conv_input = conv_input.view(-1, 1024, 7, 7)
        # 得到生成样本
        conv_output = self.model(conv_input)
        return conv_output