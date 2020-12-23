import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from Opt import opt
opt = opt()
print(opt)
class Classifier2(nn.Module):
        def __init__(self):
            super(Classifier2, self).__init__()

            def Classifier2_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.2)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.conv_blocks = nn.Sequential(
                *Classifier2_block(opt.img_channels, 16, bn=False),
                *Classifier2_block(16, 32),
                *Classifier2_block(32, 64),
                *Classifier2_block(64, 128)
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4

            # Output layers
            self.output_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax(dim=1))

        def forward(self, imgs):
            input1 = self.conv_blocks(imgs)
            input1 = input1.view(input1.shape[0], -1)
            output_predict = self.output_layer(input1)
            return output_predict