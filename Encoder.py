import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


"""
编码器
输入：带标签数据X
输出：Z分布的均值和方差
"""

class Encoder(nn.Module):
    def __init__(self, img_size, img_channels, z_size, n_classes):
        super(Encoder, self).__init__()
        
        self.img_size = img_size
        self.embed_label = spectral_norm(nn.Linear(n_classes, img_size*img_size))

        def conv_block(in_channels, out_channels, bn=True):
            layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 1))]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.25))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels, 0.8))
            return layers

        self.model = nn.Sequential(
            *conv_block(img_channels+1, 32),
            *conv_block(32, 64),
            *conv_block(64, 128),
            *conv_block(128, 256)
        )

        self.fc_mean = spectral_norm(nn.Linear(65536, z_size))
        self.fc_var = spectral_norm(nn.Linear(65536, z_size))

    def forward(self, img, label):
        embedded_label = self.embed_label(label)
        embedded_label = embedded_label.view(-1, self.img_size, self.img_size).unsqueeze(1)
        # 使得图片和标签在一起构成一个img_channels+1通道的图片
        conv_input = torch.cat([img, embedded_label], dim=1)
        # 得到一个16X16X256的张量
        conv_output = self.model(conv_input)
        # 将其打平
        conv_output = conv_output.view(conv_output.shape[0], -1)
        z_mean = self.fc_mean(conv_output)
        z_var = self.fc_var(conv_output)
        return z_mean, z_var