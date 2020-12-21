import argparse
import os
import numpy as np
import math

# PyTorch package
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
# from model import VAEGAN
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
# sklearn package for stratified sample
from sklearn.model_selection import StratifiedShuffleSplit
import random
import matplotlib
import matplotlib.pyplot as plt
from Encoder import Encoder
from Decoder import Decoder

if __name__ == "__main__":
    matplotlib.use('Agg')
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=3e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument("--lambda_ALM", type=float, default=1, help="lambda in ALM")
    parser.add_argument("--mu_ALM", type=float, default=1.2, help="mu in ALM")
    parser.add_argument("--rho_ALM", type=float, default=1.5, help="rho in ALM")
    parser.add_argument("--z_size", default=128, action="store", type=int, dest="z_size")
    parser.add_argument("--recon_level", default=3, action="store", type=int, dest="recon_level")
    parser.add_argument("--decay_lr", default=0.75, action="store", type=float, dest="decay_lr")
    parser.add_argument("--lambda_mse", default=1e-3, action="store", type=float, dest="lambda_mse")


    opt = parser.parse_args()
    print(opt)
    z_size = opt.z_size
    recon_level = opt.recon_level
    lr=opt.lr
    decay_lr=opt.decay_lr
    lambda_mse = opt.lambda_mse

    # margin and equilibirum  均衡
    margin = 0.35
    equilibrium = 0.68

    cuda = True if torch.cuda.is_available() else False
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))



    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    class VAEGAN(nn.Module):
        def __init__(self, z_size=128, recon_level=3, num_classes=10):
            super(VAEGAN, self).__init__()

            # latent space size
            self.z_size = z_size
            self.encoder = Encoder(z_size=self.z_size)
            self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size,
                                num_classes=num_classes)

            self.discriminator = Discriminator()

            # initialize self defined params
            self.init_parameters()

        def init_parameters(self):
            # just explore the network, find every weight and bias matrix and fill it
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                        # init as original implementation
                        scale = 1.0 / np.sqrt(np.prod(m.weight.shape[1:]))
                        scale /= np.sqrt(3)
                        # nn.init.xavier_normal(m.weight,1)
                        # nn.init.constant(m.weight,0.005)
                        nn.init.uniform(m.weight, -scale, scale)
                    if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                        nn.init.constant(m.bias, 0.0)

        def forward(self, ten, one_hot_class, gen_size=10):
            if self.training:
                # save original images
                ten_original = ten
                # encode
                mu, log_variances = self.encoder(ten)

                # we need true variance not log
                variances = torch.exp(log_variances * 0.5)

                # sample from gaussian
                ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)

                # shift and scale using mean and variances
                ten = ten_from_normal * variances + mu

                # decode tensor
                ten = self.decoder(ten, one_hot_class)

                # discriminator for reconstruction
                ten_layer = self.discriminator(ten, ten_original, mode='REC')

                # decode from samples
                ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)

                ten = self.decoder(ten_from_normal, one_hot_class)
                #ten_real_fake, ten_aux = self.discriminator(ten_original, ten, mode='GAN')

                #return ten, ten_real_fake, ten_layer, mu, log_variances, ten_aux
                return ten,mu,log_variances,ten_layer
            else:
                if ten is None:
                    # just sample and decode
                    ten = Variable(torch.randn(gen_size, self.z_size).cuda(), requires_grad=False)
                else:
                    mu, log_variances = self.encoder(ten)
                    # we need true variance not log
                    variances = torch.exp(log_variances * 0.5)

                    # sample from gaussian
                    ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)

                    # shift and scale using mean and variances
                    ten = ten_from_normal * variances + mu

                # decode tensor
                ten = self.decoder(ten, one_hot_class)
                return ten

        def __call__(self, *args, **kwargs):
            return super(VAEGAN, self).__call__(*args, **kwargs)

        @staticmethod
        def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original, labels_sampled,
                mu, variances, aux_labels_predicted, aux_labels_sampled, aux_labels_original):
            """
            :param ten_original: original images
            :param ten_predict: predicted images (decode ouput)
            :param layer_original: intermediate layer for original (intermediate output of discriminator)
            :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
            :param labels_original: labels for original (output of the discriminator)
            :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
            :param mu: means
            :param variances: tensor of diagonals of log_variances
            :param aux_labels_original: tensor of diagonals of log_variances
            :param aux_labels_predicted: tensor of diagonals of log_variances
            :param aux_labels_sampled: tensor of diagonals of log_variances
            :return:
            """

            # reconstruction errors, not used as part of loss just to monitor
            nle = 0.5 * (ten_original.view(len(ten_original), -1)) - ten_predict.view((len(ten_predict), -1)) ** 2

            # kl-divergence
            kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mu, 2) + variances + 1, 1)

            # mse between intermediate layers
            mse = torch.sum((layer_original - layer_predicted) ** 2, 1)

            # BCE for decoder & discriminator for original, sampled & reconstructed
            # the only excluded is the bce_gen original

            bce_dis_original = -torch.log(labels_original)
            bce_dis_sampled = -torch.log(1 - labels_sampled)

            bce_gen_original = -torch.log(1 - labels_original)
            bce_gen_sampled = -torch.log(labels_sampled)

            aux_criteron = nn.NLLLoss()
            nllloss_aux_original = aux_criteron(aux_labels_predicted, aux_labels_original)
            nllloss_aux_sampled = aux_criteron(aux_labels_sampled, aux_labels_original)

            '''
            bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                            Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
            bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                        Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
            bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
                                            Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
            bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                            Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
            bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                        Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
            '''

            return nle, kl, mse, bce_dis_original, bce_dis_sampled, bce_gen_original, bce_gen_sampled, nllloss_aux_original, nllloss_aux_sampled



    #WCVAE
    # train encoder and decoder


    # class Generator(nn.Module):
    #     def __init__(self):
    #         super(Generator, self).__init__()

    #         self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

    #         self.init_size = opt.img_size // 4  # Initial size before upsampling
    #         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

    #         self.conv_blocks = nn.Sequential(
    #             nn.BatchNorm2d(128),
    #             nn.Upsample(scale_factor=2),
    #             nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),

    #             nn.BatchNorm2d(128, 0.8),
    #             nn.LeakyReLU(0.2, inplace=True),
    #             nn.Upsample(scale_factor=2),
    #             nn.utils.spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),

    #             nn.BatchNorm2d(64, 0.8),
    #             nn.LeakyReLU(0.2, inplace=True),
    #             nn.utils.spectral_norm(nn.Conv2d(64, opt.channels, 3, stride=1, padding=1)),

    #             nn.Tanh(),
    #         )

    #     def forward(self, noise, labels):
    #         input1 = torch.mul(self.label_emb(labels), noise)
    #         input2 = self.l1(input1)
    #         input2 = input2.view(input2.shape[0], 128, self.init_size, self.init_size)
    #         output_imgs = self.conv_blocks(input2)
    #         return output_imgs


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
                *discriminator_block(opt.channels, 16, bn=False),
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
                *Classifier1_block(opt.channels, 16, bn=False),
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
            input1 = input1.view(input1.shape[0], opt.channels, opt.img_size, opt.img_size)
            input2 = torch.mul(imgs, input1)
            input3 = self.conv_blocks(input2)
            input3 = input3.view(input3.shape[0], -1)
            output_validity = self.output_layer(input3)
            return output_validity

    class Classifier2(nn.Module):
        def __init__(self):
            super(Classifier2, self).__init__()

            def Classifier2_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.2)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.conv_blocks = nn.Sequential(
                *Classifier2_block(opt.channels, 16, bn=False),
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

    # Initialize CrossEntropyLoss
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator Decoder Encoder and discriminator 
    #generator = Generator()
    net = VAEGAN(z_size=z_size, recon_level=recon_level)
    discriminator = Discriminator()
    classifier1 = Classifier1()
    classifier2 = Classifier2()

    if cuda:
        #initial net
        net.cuda()
        #generator.cuda()
        discriminator.cuda()
        classifier1.cuda()
        classifier2.cuda()
        cross_entropy_loss.cuda()

    # Initialize weights
    #generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    classifier1.apply(weights_init_normal)
    classifier2.apply(weights_init_normal)

    # Configure data loader
    os.makedirs("./data/mnist", exist_ok=True)
    # draw loss [[x],[G],[D],[C1]] and configure loss figure folder
    lossMat = [ [] for i in range(4) ]
    os.makedirs("./lossfigures", exist_ok=True)

    # -----------------------------------------------------------------------------------------
    #dateset:cifar10
    #dataset_name = 'cifar10'
    #  if dataset_name == 'cifar10':
    #         dataset = dset.CIFAR10(
    #             root=train_folder, download=True,
    #             transform=transforms.Compose([
    #                 transforms.Scale(z_size),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #             ]))
    #         dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
    #                                                  shuffle=True, num_workers=4)
    # 从MNIST中随机抽取100个数据作为带标签数据集
    MNIST = datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])]
            )
        )



    labels = [MNIST[i][1] for i in range(len(MNIST))]
    labeledset_spliter = StratifiedShuffleSplit(n_splits=1, train_size=100)
    labeled_indices, target_batch = list(labeledset_spliter.split(MNIST, labels))[0]
    labeled_MNIST = Subset(MNIST, labeled_indices)

    labeled_dataloader = DataLoader(
        labeled_MNIST,
        num_workers=8,
        pin_memory=True,
        batch_size=opt.batch_size,
        shuffle=True
    )
    all_dataloader = DataLoader(
        MNIST,
        num_workers=8,
        pin_memory=True,
        batch_size=opt.batch_size,
        shuffle=True
    )
    # ------------------------------------------------------------------------------------------

    # Optimizers
    optimizer_encoder = RMSprop(params=net.encoder.parameters(), lr=opt.lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0,
                                    centered=False)
    # lr_encoder = MultiStepLR(optimizer_encoder,milestones=[2],gamma=1)
    #Adjust learning rate
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=decay_lr)
    # optimizer_decoder = Adam(params=net.decoder.parameters(),lr = lr,betas=(0.9,0.999))
    optimizer_decoder = RMSprop(params=net.decoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0,
                                    centered=False)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=decay_lr)
    #optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
    optimizer_D = RMSprop(discriminator.parameters(), lr=0.0004)
    optimizer_C1 = RMSprop(classifier1.parameters(), lr=0.0004)
    optimizer_C2 = RMSprop(classifier2.parameters(), lr=0.0004)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # 输出生成图像
    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        #generated_labels = Variable(LongTensor(np.array([num for _ in range(n_row) for num in range(n_row)])))
        generated_imgs, mu, variances,out_layer = net(data_in, one_hot_class)
        #generated_imgs = generator(z, generated_labels)
        save_image(generated_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    # 绘制loss曲线
    def draw_loss(lossMat, batches_done):
        """Draw the loss line"""
        fig = plt.figure()
        loss_fig = fig.add_subplot(1,1,1)
        # G
        loss_fig.plot(lossMat[0], lossMat[1], 'r-', label='G loss', lw=0.5)
        # D
        loss_fig.plot(lossMat[0], lossMat[2], 'y-', label='D loss', lw=0.5)
        # C1
        loss_fig.plot(lossMat[0], lossMat[3], 'b-', label='C1 loss', lw=0.5)

        loss_fig.set_xlabel("Batches Done")
        loss_fig.set_ylabel("Loss Value")
        loss_fig.legend(loc='best')

        plt.draw()

        name = "lossfigures/Loss " + str(batches_done) + ".png"
        plt.savefig(name, dpi=300, bbox_inches='tight')

    # lambda_ALM = opt.lambda_ALM
    # mu_ALM = opt.mu_ALM

    # ----------
    #  Training
    # ----------l.
    #为了解决dataloader多线程在windows平台下出现的问题
    if __name__ == '__main__':
        for epoch in range(opt.n_epochs):
            for i, (all_imgs, target_batch) in enumerate(all_dataloader):
                # set to train mode
                net.train()

                
                batch_size = all_imgs.shape[0]
                # 得到带标签数据batch
                labeled_dataloader_iter = iter(labeled_dataloader)
                labeled_imgs, labels= next(labeled_dataloader_iter)
                # 将这些数据转换为Variable用于求导
                labeled_imgs = Variable(labeled_imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))
                
                
                #VAE数据处理，生成one-hot标签
                data_target = Variable(target_batch, requires_grad=False).float().cuda()
                data_in = Variable(all_imgs, requires_grad=False).float().cuda()
                aux_label_batch = Variable(target_batch, requires_grad=False).long().cuda()
                one_hot_class = F.one_hot(aux_label_batch).float()

                all_imgs = Variable(all_imgs.type(FloatTensor))

                # 生成噪声z和标签y
                # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                # generated_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
                
                # 生成合成图像
                #generated_imgs = generator(z, generated_labels)
                # get generated_imgs from  decoder
                generated_imgs, mu, variances,out_layer = net(data_in, one_hot_class)
                # split so we can get the different parts
                out_layer_predicted = out_layer[:len(out_layer) // 2]
                out_layer_original = out_layer[len(out_layer) // 2:]

                # selectively disable the decoder of the discriminator if they are unbalanced
                train_dis = True
                train_dec = True
                # if torch.mean(bce_dis_original_value).data < equilibrium - margin or torch.mean(
                #         bce_dis_sampled_value).data < equilibrium - margin:
                #     train_dis = False
                # if torch.mean(bce_dis_original_value).data > equilibrium + margin or torch.mean(
                #         bce_dis_sampled_value).data > equilibrium + margin:
                #     train_dec = False
                # if train_dec is False and train_dis is False:
                #     train_dis = True
                #     train_dec = True
                    
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # 甄别器计算loss
                real_validity = discriminator(all_imgs)
                fake_validity = discriminator(generated_imgs.detach())
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

                d_loss.backward()
                optimizer_D.step()
                
                # -----------------
                #  Train Classifier1
                # -----------------

                optimizer_C1.zero_grad()

                labeled_predict = classifier1(labeled_imgs, labels)
                generated_predict = classifier1(generated_imgs.detach(), one_hot_class.detach())
                c1_loss = -torch.mean(labeled_predict) + torch.mean(generated_predict)

                c1_loss.backward()
                optimizer_C1.step()

                # -----------------
                #  Train Generator
                # -----------------
                # THIS IS THE MOST IMPORTANT PART OF THE CODE
                # kl-divergence
                kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mu, 2) + variances + 1, 1)
                mse_value = torch.sum((out_layer_original - out_layer_predicted) ** 2, 1)
                loss_encoder = torch.sum(kl) + torch.sum(mse_value)
                # loss_discriminator = torch.sum(bce_dis_original_value) + torch.sum(bce_dis_sampled_value) \
                #                      + torch.sum(nllloss_aux_original) + torch.sum(nllloss_aux_sampled)

                loss_decoder = torch.sum(lambda_mse * mse_value) - d_loss
                # clean grads
                net.zero_grad()
                # encoder
                loss_encoder.backward(retain_graph=True)
                #启动优化器
                optimizer_encoder.step()
                # clean others, so they are not afflicted by encoder loss
                net.zero_grad()
                # decoder
                loss_decoder.backward(retain_graph=True)
                optimizer_decoder.step()
                # clean the discriminator
                # net.discriminator.zero_grad()
                
                #optimizer_G.zero_grad()

                # real_validity_G = discriminator(all_imgs)
                # fake_validity_G = discriminator(generated_imgs)
                # generated_predict_G = classifier1(generated_imgs, generated_labels)
                # L_d = - torch.mean(fake_validity_G)
                # L_c1 = - torch.mean(generated_predict_G)
                # g_loss =  L_c1 / 2 + L_d / 2
                # g_loss.backward()
                # optimizer_G.step()
                
                # -----------------
                #  Train Classifier2
                # -----------------

                optimizer_C2.zero_grad()
                # get generated_imgs from  decoder
                generated_imgs_C2, mu, variances,out_layer = net(data_in, one_hot_class)
                #generated_imgs_C2 = generator(z, generated_labels)
                predict_C2 = classifier2(generated_imgs_C2)
                c2_loss = cross_entropy_loss(predict_C2, one_hot_class)

                c2_loss.backward()
                optimizer_C2.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [C1 loss: %f] [G loss: %f] [C2 loss on G-img: %f]"
                    % (epoch, opt.n_epochs, i, len(all_dataloader), d_loss.item(), c1_loss.item(),loss_encoder.item(),loss_decoder.item(), )
                )
                
                batches_done = epoch * len(all_dataloader) + i

                lossMat[0].append(batches_done)
                lossMat[1].append(loss_encoder.item())
                lossMat[1].append(loss_decoder.item())
                lossMat[2].append(d_loss.item())
                lossMat[3].append(c1_loss.item())

                if batches_done % opt.sample_interval == 0:
                    sample_image(n_row=10, batches_done=batches_done)
                    draw_loss(lossMat, batches_done)
            lr_encoder.step()
            lr_decoder.step()
            # generate_imgs_L = generator(z, generate_labels.detach())
            # fake_validity_L = discriminator(generate_imgs_L)
            # real_validity_L = discriminator(real_imgs.detach())
            # L = torch.mean(real_validity_L) - torch.mean(fake_validity_L)
            # lambda_ALM = lambda_ALM + mu_ALM * L.detach()
