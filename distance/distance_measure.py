import sys
sys.path.append("..")

from models import *
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class GetDistancePGD:
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.epsilon = torch.tensor(np.inf)
        self.clip_max = torch.tensor(np.inf)
        self.clip_min = torch.tensor(-np.inf)
        self.max_steps = 100 # if not specify, the default setting is 100
        self.step_size=0.1/255 # if not specify, the default setting is 0.1/255

    def calculate_distance(self, train_loader):
        if self.args.dataset == 'cifar10':
            concatenated_tensor = torch.empty((0, 3, 32, 32))
        else:
            raise NotImplementedError
        for x, y in train_loader:
            for inbatch_idx in range(x.shape[0]):
                x_adv = self.pgd_attack(x[inbatch_idx].unsqueeze(0).to(self.device),y[inbatch_idx].unsqueeze(0).to(self.device))
                diff_tensor = x_adv - x[inbatch_idx].unsqueeze(0).to(self.device)
                concatenated_tensor = torch.cat((concatenated_tensor, diff_tensor.detach().cpu()), dim=0)
        torch.save(concatenated_tensor, self.args.save_path)

    def pgd_attack(self, x, y):
        x_adv = x.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        
        output = self.model(x_adv)
        y = self.model(x_adv).argmax(1)
        for i in range(self.max_steps):
            self.model.zero_grad()
        
            output = self.model(x_adv)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = x_adv.grad.data.detach()
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv.requires_grad_()
            result = self.judge_predict(self.model, x_adv, y)

            if result:
                break
            
        return x_adv
        
    def judge_predict(self, model, x_adv, y):
        y_pgd = model(x_adv).argmax(1)
        if y_pgd == y:
            return False
        return True
