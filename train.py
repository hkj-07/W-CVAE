import os
import numpy as np
import math
import random

# PyTorch package
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

# sklearn分割数据
from sklearn.model_selection import StratifiedShuffleSplit

# 画图包
import matplotlib
import matplotlib.pyplot as plt

# 导入需要的网络和参数模块
from Opt import Opt
from Encoder import Encoder
from Decoder import Decoder
from Discriminator_X import Discriminator_X
from Discriminator_Z import Discriminator_Z
from Classifier import Classifier

# 定义初始化权重的函数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 检测是否可以GPU加速
cuda = True if torch.cuda.is_available() else False
print(torch.cuda.is_available())
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

# 设置需要的tensor格式
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# 交叉熵
cross_entropy = torch.nn.CrossEntropyLoss()

# 实例化子网络与参数
opt = Opt()
encoder = Encoder()
decoder = Decoder()
discriminator_x = Discriminator_X()
discriminator_z = Discriminator_Z()
classifier = Classifier()

# 转gpu加速
if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator_x.cuda()
    discriminator_z.cuda()
    classifier.cuda()
    cross_entropy.cuda()

# 权重初始化
encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
discriminator_x.apply(weights_init_normal)
discriminator_z.apply(weights_init_normal)
classifier.apply(weights_init_normal)

matplotlib.use('Agg')

# 设置生成图像输出文件夹
os.makedirs("./images", exist_ok=True)
# 设置loss曲线图输出文件夹
os.makedirs("./lossfigures", exist_ok=True)
# 设置数据集文件夹
os.makedirs("./data/mnist", exist_ok=True)

"""
loss矩阵
[[迭代次数],[encoder loss],[decoder loss],[discriminator_z loss],[discriminator_x loss],[classifier loss]] 
"""
lossMat = [ [] for i in range(6) ]

# 加载数据集
MNIST = datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])]
        )
    )
# 分割数据集
labels = [MNIST[i][1] for i in range(len(MNIST))]
labeledset_spliter = StratifiedShuffleSplit(n_splits=1, train_size=100)
labeled_indices, target_batch = list(labeledset_spliter.split(MNIST, labels))[0]
labeled_MNIST = Subset(MNIST, labeled_indices)
# 设置dataloader
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

# 设置Optimizers
optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_discriminator_x = torch.optim.Adam(discriminator_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_discriminator_z = torch.optim.Adam(discriminator_z.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_classifier = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 输出生成图像
def sample_image(batches_done):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (10*10, opt.z_size))))
    # Get labels ranging from 0 to n_classes for n rows
    generated_labels = LongTensor(np.array([num for _ in range(10) for num in range(10)]))
    generated_labels = F.one_hot(generated_labels)
    generated_labels = Variable(generated_labels.type(FloatTensor))
    generated_imgs = decoder(z, generated_labels)
    # 保存图片
    save_image(generated_imgs.data, "./images/%d.png" % batches_done, nrow=10, normalize=True)

# 绘制loss曲线
# def draw_loss(lossMat, batches_done):
#     """Draw the loss line"""
#     fig = plt.figure()
#     loss_fig = fig.add_subplot(1,1,1)
#     # G
#     loss_fig.plot(lossMat[0], lossMat[1], 'r-', label='G loss', lw=0.5)
#     # D
#     loss_fig.plot(lossMat[0], lossMat[2], 'y-', label='D loss', lw=0.5)
#     # C1
#     loss_fig.plot(lossMat[0], lossMat[3], 'b-', label='C1 loss', lw=0.5)

#     loss_fig.set_xlabel("Batches Done")
#     loss_fig.set_ylabel("Loss Value")
#     loss_fig.legend(loc='best')

#     plt.draw()

#     name = "lossfigures/Loss " + str(batches_done) + ".png"
#     plt.savefig(name, dpi=300, bbox_inches='tight')


"""
训练模型
"""
#为了解决dataloader多线程在windows平台下出现的问题
if __name__ == '__main__':

    for epoch in range(opt.n_epochs):

        for i, (imgs, _) in enumerate(all_dataloader):

            # 得到带标签数据 转one-hot
            labeled_dataloader_iter = iter(labeled_dataloader)
            labeled_imgs, labels= next(labeled_dataloader_iter)
            target = labels
            labels = F.one_hot(labels)

            # 将这些数据转换为Variable用于求导
            labels = Variable(labels.type(FloatTensor))
            labeled_imgs = Variable(labeled_imgs.type(FloatTensor))
            imgs = Variable(imgs.type(FloatTensor))
            target = Variable(target.type(LongTensor))

            """
            训练discriminator for z
            """
            optimizer_discriminator_z.zero_grad()

            z_mean, z_var = encoder(labeled_imgs, labels)
            z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.z_size))))
            z = torch.add(torch.mul(z, z_var), z_mean)
            validity_z = discriminator_z(z)
            z_normal = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.z_size))))
            validity_z_normal = discriminator_z(z_normal)
            discriminator_z_loss = -torch.mean(validity_z_normal) + torch.mean(validity_z)
            discriminator_z_loss.backward()

            optimizer_discriminator_z.step()
            
            """
            训练encoder
            """
            optimizer_encoder.zero_grad()
            
            z_mean, z_var = encoder(labeled_imgs, labels)
            z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.z_size))))
            z = torch.add(torch.mul(z, z_var), z_mean)
            validity_z = discriminator_z(z)
            encoder_loss = -torch.mean(validity_z)
            encoder_loss.backward()
            
            optimizer_encoder.step()

            """
            训练discriminator for x
            """
            optimizer_discriminator_x.zero_grad()

            z_mean, z_var = encoder(labeled_imgs, labels)
            z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.z_size))))
            z = torch.add(torch.mul(z, z_var), z_mean)
            generated_imgs = decoder(z, labels)
            validity_generated_imgs = discriminator_x(generated_imgs)
            validity_imgs = discriminator_x(imgs)
            discriminator_x_loss = -torch.mean(validity_imgs) + torch.mean(validity_generated_imgs)
            discriminator_x_loss.backward()

            optimizer_discriminator_x.step()
            
            """
            训练decoder
            """
            optimizer_decoder.zero_grad()

            z_mean, z_var = encoder(labeled_imgs, labels)
            z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.z_size))))
            z = torch.add(torch.mul(z, z_var), z_mean)
            generated_imgs = decoder(z, labels)
            validity_generated_imgs = discriminator_x(generated_imgs)
            decoder_loss = F.mse_loss(generated_imgs, labeled_imgs) - opt.lambda_ALM * torch.mean(validity_generated_imgs)
            decoder_loss.backward()

            optimizer_decoder.step()

            """
            训练classifier
            """
            optimizer_classifier.zero_grad()

            z_mean, z_var = encoder(labeled_imgs, labels)
            z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.z_size))))
            z = torch.add(torch.mul(z, z_var), z_mean)
            generated_imgs = decoder(z, labels)
            predicts = classifier(generated_imgs.detach())
            classifier_loss = cross_entropy(predicts, target)
            classifier_loss.backward()

            optimizer_classifier.step()

            # 控制台输出loss
            print(
                "[Epoch %d/%d] [Batch %d/%d] [encoder loss: %f] [decoder loss: %f] [discri_z loss: %f] [discri_x loss: %f] [classifier loss: %f]"
                % (epoch, opt.n_epochs, i, len(all_dataloader), encoder_loss.item(), decoder_loss.item(), discriminator_z_loss.item(), discriminator_x_loss.item(), classifier_loss.item())
            )
            
            batches_done = epoch * len(all_dataloader) + i

            lossMat[0].append(batches_done)
            lossMat[1].append(encoder_loss.item())
            lossMat[2].append(decoder_loss.item())
            lossMat[3].append(discriminator_z_loss.item())
            lossMat[4].append(discriminator_x_loss.item())
            lossMat[5].append(classifier_loss.item())

            if batches_done % opt.sample_interval == 0:
                sample_image(batches_done=batches_done)
                # draw_loss(lossMat, batches_done)
        
        # generate_imgs_L = generator(z, generate_labels.detach())
        # fake_validity_L = discriminator(generate_imgs_L)
        # real_validity_L = discriminator(real_imgs.detach())
        # L = torch.mean(real_validity_L) - torch.mean(fake_validity_L)
        # lambda_ALM = lambda_ALM + mu_ALM * L.detach()