import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pickle


class GetLoss:
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args

    def calculate_loss(self, train_loader):
        loss_list = []
        for x, y, idx in train_loader:
            for inbatch_idx in range(x.shape[0]):
                x_adv = self.get_loss(x[inbatch_idx].unsqueeze(0).to(self.device),y[inbatch_idx].unsqueeze(0).to(self.device))
                loss_list.append(x_adv)
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(loss_list, f)

    def get_loss(self, x, y):
        x_adv = x.clone().detach().requires_grad_(True)
        self.model.eval()        
        with torch.no_grad():
            outputs = self.model(x_adv)
            clean_loss = F.cross_entropy(outputs, y)
        return clean_loss.item()
        
