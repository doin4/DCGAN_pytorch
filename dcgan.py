# @SHU-doin4
# 2021.12.29
# practice
import torch 
import torch.nn as nn
import torchvision 
import os
import argparse
import ResNet as doin4
import numpy as np
# parameter helper

from torch.nn import functional as F 
from torch.utils import data 
from torchvision import transforms
from torch.autograd import Variable

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100000, help="set training epoch number")
    parser.add_argument("--batch_size", type=int, default=1, help="set batch_size")
    parser.add_argument("--g_lr", type=float, default=0.0004, help="set net learing rate")   
    parser.add_argument("--d_lr", type=float, default=0.0002, help="set net learing rate")    
    parser.add_argument("--d1", type=float, default=0.5, help="set first adam decay rate")
    parser.add_argument("--d2", type=float, default=0.999, help="set secend adam decay rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="set the demension of latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of the image")
    parser.add_argument("--channels", type=int, default=3, help="channels of the image, check every time you change image")
    parser.add_argument("--visualize", type=int, default=1000, help="check generated image")
    parser.add_argument("--save_path", type=str, default="./output", help="name of the save file")
    parser.add_argument("--img_path", type=str, default="./image", help="name of the save file")
    parser.add_argument("--train_k", type=int, default=2, help="every k time, update D_net's gradient")
    opt = parser.parse_args()
    return opt

class generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # note: the input of the generator is random noise
        # --//-- is to ensure get a int
        # when passing data in pytorch, the format of the image is [batch_size, channels, width, high]
        # init_size is to set the real conv_block's input shape
        self.init_size =  opt.img_size // 4
        self.layer1 = nn.Linear(in_features=opt.latent_dim, out_features=128 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            *doin4.resnet_block(input_channels=128, num_channels=128, num_residuals=3, first_block=True),
            nn.Upsample(scale_factor=2),
            *doin4.resnet_block(input_channels=128, num_channels=128, num_residuals=3, first_block=True),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        z = self.layer1(z)
        # have to change the shape of x, to fit the whole net
        # this is matched with the output of the linear
        return self.conv_blocks(z.view(z.shape[0], 128, self.init_size, self.init_size))

class discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.channels), nn.Sigmoid())
        
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# if you want to try your own data, then use this one
def get_iter(opt):
    date_iter = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(opt.img_path,
                transform=transforms.Compose(
                    [transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
                ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    return date_iter

# uncomment this if you wan to use CIFAR10 or FASHION-MNIST dataset 
# def get_iter():
#     date_iter = torch.utils.data.DataLoader(
#         torchvision.datasets.CIFAR10(
#                 "../datasets",
#                 train=True,
#                 download=False,
#                 transform=transforms.Compose(
#                     [transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#                 ),
#             batch_size=opt.batch_size,
#             shuffle=True,
#         )
#     return date_iter

def train(opt):
    g_net = generator(opt).to('cuda')
    d_net = discriminator(opt).to('cuda')
    adversarial_loss = torch.nn.BCELoss().to('cuda')
    g_net.apply(init_weights)
    d_net.apply(init_weights)
    data_iter = get_iter(opt)
    optimizer_G = torch.optim.Adam(g_net.parameters(), lr=opt.g_lr, betas=(opt.d1, opt.d2))
    optimizer_D = torch.optim.Adam(d_net.parameters(), lr=opt.d_lr, betas=(opt.d1, opt.d2))
    Tensor = torch.cuda.FloatTensor     
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(data_iter):

            # adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], opt.channels).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], opt.channels).fill_(0.0), requires_grad=False)

            # configure input
            real_imgs = Variable(imgs.type(Tensor))

            #  train G_net
            optimizer_G.zero_grad()

            # noise input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # generated image
            gen_imgs = g_net(z)

            # modified loss function, which can be found in "improved DCGAN"
            g_loss = adversarial_loss(d_net(gen_imgs), valid) ** 2   #

            g_loss.backward()
            optimizer_G.step()

            #  train D_net
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(d_net(real_imgs), valid)
            fake_loss = adversarial_loss(d_net(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            batches_done = epoch * len(data_iter) + i
            if batches_done % opt.train_k == 0:
                d_loss.backward()
                optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(data_iter), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.visualize == 0:
                # the totle generate image numbuer is the batch_size
                # if you have more input image, them you can get more generate image
                # if you want to show n images, just change the gen_imgs.data -> gen_imgs.data[:n]
                path = opt.save_path + "/%d" % batches_done + ".png"
                torchvision.utils.save_image(gen_imgs.data, path, nrow=2, normalize=True)

if __name__ == '__main__':
    opt = get_opt()
    os.makedirs(opt.save_path, exist_ok=True)
    train(opt)