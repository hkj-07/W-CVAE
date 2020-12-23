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
from model import VAEGAN
from Discriminator import Discriminator
from Classifier1 import Classifier1
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

# sklearn package for stratified sample
from sklearn.model_selection import StratifiedShuffleSplit
import random
import matplotlib
import matplotlib.pyplot as plt

"""
导入初始化参数
"""
from Opt import opt

if __name__ == "__main__":
    matplotlib.use('Agg')
    os.makedirs("images", exist_ok=True)
    opt = opt()
    print(opt)
    z_size = opt.z_size
    
    lr=opt.lr
    decay_lr=opt.decay_lr
    lambda_mse = opt.lambda_mse

    # margin and equilibirum  均衡
    # margin = 0.35
    # equilibrium = 0.68

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


    # Initialize CrossEntropyLoss
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator Decoder Encoder and discriminator 
    #generator = Generator()
    net = VAEGAN(z_size=z_size)
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