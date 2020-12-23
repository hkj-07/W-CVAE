import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from Opt import opt
opt = opt()
print(opt)

class Classifier1(nn.Module):
        def __init__(self):
            super(Classifier1, self).__init__()

            self.label_emb = nn.Embedding(opt.n_classes, (opt.img_size ** 2) * opt.channels)

            def Classifier1_block(in_filters, out_filters, bn=True):
                
                block = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.2)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.conv_blocks = nn.Sequential(
                *Classifier1_block(opt.img_channels, 16, bn=False),
                *Classifier1_block(16, 32),
                *Classifier1_block(32, 64),
                *Classifier1_block(64, 128)
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4

            # Output layers
            self.output_layer = nn.Sequential(nn.utils.spectral_norm(nn.Linear(128 * ds_size ** 2, 1)))

        def forward(self, imgs, labels):
            input1 = self.label_emb(labels)
            input1 = input1.view(input1.shape[0], opt.img_channels, opt.img_size, opt.img_size)
            input2 = torch.mul(imgs, input1)
            input3 = self.conv_blocks(input2)
            input3 = input3.view(input3.shape[0], -1)
            output_validity = self.output_layer(input3)
            return output_validity