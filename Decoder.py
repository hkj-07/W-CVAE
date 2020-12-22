import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

"""
解码器
输入：Z和标签
输出：对应类别的x
"""
        
class Decoder(nn.Module):
    def __init__(self, img_size, img_channels, z_size, n_classes):
        super(Decoder, self).__init__()

        self.img_size = img_size
        self.embed_z = spectral_norm(nn.Linear(z_size+n_classes, 7*7*256))

        def conv_block(in_channels, out_channels, bn=True, up=True):
            layers = []
            if up:
                # 每次上采样，图像边长翻倍 7 14 28 只能翻倍两次
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels, 0.8))
            return layers
        
        self.model = nn.Sequential(
            *conv_block(256, 128),
            *conv_block(128, 64),
            *conv_block(64, img_channels, up=False)
        )

    def forward(self, z, label):
        # 将z和label连起来
        catted_z = torch.cat([z, label], dim=1)
        # 映射到256*7*7的图片
        conv_input = self.embed_z(catted_z)
        conv_input = conv_input.view(-1, 256, self.img_size, self.img_size)
        # 得到生成样本
        conv_output = self.model(conv_input)
        return conv_output