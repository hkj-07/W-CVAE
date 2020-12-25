import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from Opt import Opt

"""
编码器
输入：带标签数据X
输出：Z
"""

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.opt = Opt()
        
        # self.embed_label = spectral_norm(nn.Linear(self.opt.n_classes, self.opt.img_size*self.opt.img_size))

        def conv_block(in_channels, out_channels, bn=True):
            layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=1))]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.25))
            return layers

        self.model = nn.Sequential(
            *conv_block(self.opt.img_channels, 64),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512)
        )

        self.fc_mean = spectral_norm(nn.Linear(16*16*512, self.opt.z_size))
        # self.fc_var = spectral_norm(nn.Linear(16*16*512, self.opt.z_size))

    def forward(self, img, label):
        # embedded_label = self.embed_label(label)
        # embedded_label = embedded_label.view(-1, self.opt.img_size, self.opt.img_size).unsqueeze(1)
        # 使得图片和标签在一起构成一个img_channels+1通道的图片
        # conv_input = torch.cat([img, embedded_label], dim=1)
        # 得到一个16X16X256的张量
        conv_output = self.model(img)
        # 将其打平
        conv_output = conv_output.view(conv_output.shape[0], -1)
        z = self.fc_mean(conv_output)
        # z_var = self.fc_var(conv_output)
        return z