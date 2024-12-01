'''
This file contains the reconstruction attack based on optimization process
code is modified from https://github.com/NVlabs/DeepInversion
'''

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random
import numpy as np
import os


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

class DeepInversionFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

class OptimizationBase(object):
    def __init__(self, args, 
                 target_model=None, 
                 bs=200,
                 lr=0.2,
                 iteration_num=1000,
                 path="./gen_images/",
                 final_data_path="/gen_images_final/",
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 coefficients=dict(),
                 network_output_function=lambda x: x):
        torch.manual_seed(torch.cuda.current_device())

        self.args = args
        self.bs = bs
        self.target_model = target_model.to(args.device)
        self.input_shape = args.input_shape
        self.criterion = criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.iteration_num = iteration_num
        self.do_flip = False #True unless the images are not normalized to [0,1]
        self.num_generations = 0

        self.bn_reg_scale = 0.05
        self.first_bn_multiplier = 10.
        self.var_scale_l1 = 0.0
        self.var_scale_l2 = 0.0001
        self.l2_scale = 0.00001
        self.lr = 0.2
        self.main_loss_multiplier = 1.0
        self.adi_scale = 0

        self.loss_r_feature_layers = []

        for module in self.target_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    def get_images(self, targets=None):
        save_every = 100
        target_model = self.target_model
        best_cost = 1e4
        criterion = self.criterion

        if targets is None:
            targets = [i for i in range(self.args.num_class)]
            targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to(self.args.device)

        data_type = torch.float
        inputs = torch.randn((self.bs, *self.input_shape), requires_grad=True, device=self.args.device,
                             dtype=data_type)

        lim_0, lim_1 = 2, 2
        iteration = 0
        optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
        do_clip = True

        lr_scheduler = lr_cosine_policy(self.lr, 100, self.iteration_num)

        for iteration_loc in range(self.iteration_num):
            iteration += 1
            lr_scheduler(optimizer, iteration_loc, iteration_loc)
            inputs_jit = inputs

            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            flip = random.random() > 0.5
            if flip and self.do_flip:
                inputs_jit = torch.flip(inputs_jit, dims=(3,))

            optimizer.zero_grad()
            target_model.zero_grad()
            outputs = target_model(inputs_jit)
            loss = criterion(outputs, targets)

            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

            rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers)-1)]
            loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])

            loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()
            if self.args.attack_type == 'deepinversion':
                loss_aux = self.var_scale_l2 * loss_var_l2 + \
                            self.var_scale_l1 * loss_var_l1 + \
                            self.bn_reg_scale * loss_r_feature + \
                            self.l2_scale * loss_l2

            loss = self.main_loss_multiplier * loss + loss_aux

            loss.backward()

            optimizer.step()
            if do_clip:
                inputs.data = torch.clamp(inputs.data,0,1)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()
                best_cost = loss.item()

        self.save_images(best_inputs, targets)

    def save_images(self, images, targets):
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

    def generate_batch(self, total_num, net_student=None, targets=None):
        target_model = self.target_model
        num_epoch = int(total_num/self.bs)
        for i in range(num_epoch):
            self.get_images()
            target_model.eval()
            self.num_generations += 1


        