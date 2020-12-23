import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from Opt import opt
opt = opt()
print(opt)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), 
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.2)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.img_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.output_layer = nn.Sequential(nn.utils.spectral_norm(nn.Linear(128 * ds_size ** 2, 1)))

    def forward(self, img,other_ten,mode='REC'):
        if mode == 'REC':
            for i, layer in enumerate(self.conv):
                # take 9th layer as one of the outputs
                ten, layer_ten = layer(ten, True)
                # fetch the layer representations just for the original & reconstructed,
                # flatten, because it is multidimensional
                layer_ten = layer_ten.view(len(layer_ten), -1)
                return layer_ten
                
        else:
            input1 = self.conv_blocks(img)
            input1 = input1.view(input1.shape[0], -1)
            output_validity = self.output_layer(input1)
            return output_validity