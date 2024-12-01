'''
This file contains the reconstruction attack based on paper ``The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks''
code is modified from https://github.com/MKariya1998/GMI-Attack
'''
import os
import time
import random
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
from generate_arch import GeneratorCIFAR10, DGWGANCIFAR10


def save_tensor_images(images, filename, nrow = None, normalize = True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)


def gradient_penalty(x, y, DG, device):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(device)
    z = x + alpha * (y - x)
    z = z.to(device)
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).to(device), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)
        
class Revealer(object):
    def __init__(self,args,target_model,dataloader,bs=100):
        self.args = args
        self.target_model = target_model.to(args.device)
        self.dataloader = dataloader
        self.bs = bs
        self.num_generations = 0

    def train_gan(self,G,DG,dataloader,epochs=280,lr=2e-4,z_dim=100):
        n_critic = 5
        G.to(self.args.device)
        DG.to(self.args.device)
    
        dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
        g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

        step = 0

        for epoch in range(epochs):
            for i, (imgs,labels) in enumerate(dataloader):
                step += 1
                imgs = imgs.to(self.args.device)
                bs = imgs.size(0)
                
                freeze(G)
                unfreeze(DG)

                z = torch.randn(bs, z_dim).to(self.args.device)

                f_imgs = G(z)
                r_logit = DG(imgs)
                f_logit = DG(f_imgs)
                
                wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
                gp = gradient_penalty(imgs.data, f_imgs.data, DG, self.args.device)
                dg_loss = - wd + gp * 10.0
                
                dg_optimizer.zero_grad()
                dg_loss.backward()
                dg_optimizer.step()
                if step % n_critic == 0:
                    freeze(DG)
                    unfreeze(G)
                    z = torch.randn(bs, z_dim).to(self.args.device)
                    f_imgs = G(z)
                    logit_dg = DG(f_imgs)
                    # calculate g_loss
                    g_loss = - logit_dg.mean()
                    
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

            if (epoch+1) % 10 == 0:
                z = torch.randn(32, z_dim).to(self.args.device)
                fake_image = G(z)
                save_tensor_images(fake_image.detach(), os.path.join(self.args.result_path, "result_image_{}.png".format(epoch)), nrow = 8)
            
            torch.save(G.state_dict(), GENERATOR_PATH)
            torch.save(DG.state_dict(), DISCRIMINATOR_PATH)

    def set_gan(self,z_dim=100):
        if self.args.dataset == 'cifar10':
            G = GeneratorCIFAR10(z_dim)
            DG = DGWGANCIFAR10(in_dim=3, dim=32)
        return G, DG

    def set_random_seed(self, manualSeed=None):
        if manualSeed is None:
            manualSeed = random.randint(1, 10000)    
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        cudnn.benchmark = True
    
    def inversion(self,G, D, T, iden, random_seed, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1):
        if self.args.dataset == 'mnist':
            lr=1e-2
        G.to(self.args.device)
        D.to(self.args.device)
        torch.manual_seed(random_seed) 
        torch.cuda.manual_seed(random_seed) 
        np.random.seed(random_seed) 
        random.seed(random_seed)

        criterion = nn.CrossEntropyLoss().to(self.args.device)
        bs = iden.shape[0]
        
        G.eval()
        D.eval()
        T.eval()

        max_score = torch.zeros(bs)
        max_iden = torch.zeros(bs)
        z_hat = torch.zeros(bs, 100)
        flag = torch.zeros(bs)

        z = torch.randn(bs, 100).to(self.args.device).float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).to(self.args.device).float()
            
        for i in range(iter_times):
            fake = G(z)
            label = D(fake)
            out = T(fake)
            
            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()
            Iden_Loss = criterion(out, iden)
            Total_Loss = Prior_Loss + lamda * Iden_Loss

            Total_Loss.backward()
            
            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + ( - momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()
    
        fake = G(z)
        self.save_images(fake, iden)

    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if not os.path.isdir('{}/rec_img'.format(self.args.result_path)):
                os.makedirs('{}/rec_img'.format(self.args.result_path))
            place_to_store = '{}/rec_img/class_{:03d}_numgen_{:05d}_id{:03d}.jpg'.format(self.args.result_path, class_id,
                                                                                          self.num_generations, id)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            if self.args.dataset == 'mnist':
                image_np = image_np.squeeze(2)
                pil_image = Image.fromarray((image_np * 255).astype(np.uint8),'L')
            else:
                pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self, total_num,G,DG):
        target_model = self.target_model
        target_model.eval()
        num_epoch = int(total_num/self.bs)
        targets = [i for i in range(self.args.num_class)]
        targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to(self.args.device)

        for i in range(num_epoch):
            self.inversion(G, DG, target_model, targets, num_epoch, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1)
            self.num_generations += 1

    def reconstruct_images(self,total_num):
        self.set_random_seed()
        G, DG = self.set_gan()
        if not os.path.exists(GENERATOR_PATH):
            self.train_gan(G, DG, self.dataloader)
        else:
            print('load GAN')
            G.load_state_dict(torch.load(GENERATOR_PATH))
            DG.load_state_dict(torch.load(DISCRIMINATOR_PATH))
        self.generate_batch(total_num,G,DG)