import os
import time
import torch
import argparse


import torch.optim as optim
import numpy as np
import torch.nn as nn
from torchvision.models import vgg19
from torchvision import transforms
from torch.autograd import Variable
from torchvision import datasets

import matplotlib.pyplot as plt
import network, utils

parser = argparse.ArgumentParser()

parser.add_argument('--num_epoch', type=int, default=100,
                    help='num of training epoch')
parser.add_argument('--init_num_epoch', type=int, default=1,
                    help='num of initialization epoch')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--root_path', default=1, help='batch size')
parser.add_argument('--residual_num', type=int,
                    default=8, help='residual number')
parser.add_argument('--lr_G', type=float, default=0.001,
                    help='learning rate of Generator')
parser.add_argument('--lr_D', type=float, default=0.001,
                    help='learning rate of Discriminator')
parser.add_argument('--gamma_G', type=float, default=1, help='gamma_G')
parser.add_argument('--gamma_D', type=float, default=1, help='gamma_D')
parser.add_argument('--beta_1', type=float, default=0.5, help='beta_1')
parser.add_argument('--beta_2', type=float, default=0.99, help='beta_2')
parser.add_argument('--cont_lambda', type=int, default=10, help='cont_lambda')
parser.add_argument('--load_model', type=bool, default=False, help='load previous model')

opt = parser.parse_args()

G = network.generator(residual_num=opt.residual_num)
D = network.discriminator()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)

G_optimizer = optim.Adam(G.parameters(), lr = opt.lr_G,
                         betas=(opt.beta_1, opt.beta_2))
D_optimizer = optim.Adam(D.parameters(), lr = opt.lr_D,
                         betas=(opt.beta_1, opt.beta_2))

G.to(device)
D.to(device)

vgg = vgg19(pretrained=False)
vgg.load_state_dict(torch.load('./vgg19.pth'))
vgg.to(device)
vgg.eval()

transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

src_path = "/home/zhui/CartoonGAN_pytorch/data/real/train"
cart_path = "/home/zhui/CartoonGAN_pytorch/data/violet/train"
cart_smooth_path = "/home/zhui/CartoonGAN_pytorch/data/edge_smoothed"

# if not os.path.isdir(cart_smooth_path):
#     os.mkdir(cart_smooth_path)
#     utils.edge_promoting(src_path, cart_smooth_path)
    


src_loader = utils.data_load('/home/zhui/CartoonGAN_pytorch/data/real/', 'train', transform, opt.batch_size, shuffle=True, drop_last=True)
cartoon_loader = utils.data_load('/home/zhui/CartoonGAN_pytorch/data/violet/', 'train', transform, opt.batch_size, shuffle=True, drop_last=True)
cartoon_smooth_loader = utils.data_load('/home/zhui/CartoonGAN_pytorch/data/', 'edge_smoothed', transform, opt.batch_size, shuffle=True, drop_last=True)



init_content_losses = []
D_losses = []
G_losses = []
Cont_losses = []


def pretrain():
    print("start pretrain")
    for epoch in range(opt.init_num_epoch):
        for i, img in enumerate(zip(src_loader, cartoon_loader)):
            if i > 0:
                break
            src, cart = img[0][0], img[1][0]
            src = src.to(device)
            cart = cart.to(device)

            gen_cart = G(src)

            feature = vgg((src + 1) / 2)
            G_feature = vgg((gen_cart + 1) / 2)
            Cont_loss = opt.cont_lambda * L1_loss(G_feature, feature.detach())

            G_loss = Cont_loss
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            Cont_losses.append(Cont_loss.item())

            if i % 50 == 0:
                print("i: " , i , "Content_loss: " ,Cont_loss.item())
                result = torch.cat((src[0], gen_cart[0]), 2)
                result = (result.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2
                filename = "during_pretrain_%s_%s.png" % (epoch, i)
                path = os.path.join("./result", filename)
                plt.imsave(path, result)

        average_cont_loss = np.mean(Cont_losses)

        print("epoch: " , epoch, "Content_loss: " ,average_cont_loss)

        if not os.path.isdir('models/'):
                os.mkdir('models/')

        save_path = os.path.join('models/', "pretrain-model" + str(epoch) + ".ckpt")
        torch.save({
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'G_optim_state': G_optimizer.state_dict(),
                'D_optim_state': D_optimizer.state_dict(),
            }, save_path)

    
def load_model(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state'])
    D.load_state_dict(checkpoint['D_state'])
    G_optimizer.load_state_dict(checkpoint['G_optim_state'])
    D_optimizer.load_state_dict(checkpoint['D_optim_state'])



def train():

    print("start training")
    if opt.load_model:
        load_model(os.path.join('models/', "model.ckpt"))

    for epoch in range(opt.num_epoch):

        for i, img in enumerate(zip(src_loader, cartoon_loader, cartoon_smooth_loader)):
            
            src, cart, cart_smooth = img[0][0], img[1][0], img[2][0]
            src = src.to(device)
            cart = cart.to(device)
            cart_smooth = cart_smooth.to(device)

            D_real = D(cart)
            D_real_loss = BCE_loss(D_real, Variable(torch.ones(D_real.size()).to(device)))

            gen_cart = G(src)
            D_fake = D(gen_cart)
            D_fake_loss = BCE_loss(D_fake, Variable(torch.zeros(D_fake.size()).to(device)))

            D_fake_smooth = D(cart_smooth)
            D_fake_smooth_loss = BCE_loss(D_fake_smooth, Variable(torch.zeros( D_fake_smooth.size()).to(device)  ))

            D_loss = D_real_loss + D_fake_loss + D_fake_smooth_loss

            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            gen_cart = G(src)
            D_fake = D(gen_cart)
            D_fake_loss = BCE_loss(D_fake, Variable(torch.ones( D_fake.size()).to(device)))

            feature = vgg((src + 1) / 2)
            G_feature = vgg((gen_cart + 1) / 2)
            Cont_loss = opt.cont_lambda * L1_loss(G_feature, feature.detach())

            G_loss = D_fake_loss + Cont_loss
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())
            Cont_losses.append(Cont_loss.item())

            if i % 50 == 0:
                print("i: " , i , "G_loss: ", G_loss.item() , "D_loss: " , D_loss.item() , "Content_loss: " ,Cont_loss.item())
                result = torch.cat((src[0], gen_cart[0]), 2)
                result = (result.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2
                filename = "during_train_%s_%s.png" % (epoch, i)
                path = os.path.join("./result", filename)
                plt.imsave(path, result)

        average_D_loss = np.mean(D_losses)
        average_G_loss = np.mean(G_losses)
        average_cont_loss = np.mean(Cont_losses)

        print("epoch: " , epoch , "G_loss: ", average_G_loss , "D_loss: " , average_D_loss , "Content_loss: " ,average_cont_loss)

        if not os.path.isdir('models/'):
                os.mkdir('models/')

        save_path = os.path.join('models/', "model" + str(epoch) + ".ckpt")
        torch.save({
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'G_optim_state': G_optimizer.state_dict(),
                'D_optim_state': D_optimizer.state_dict(),
            }, save_path)

        
def main():
    pretrain()
    train()

if __name__ == "__main__":
    main()
    
