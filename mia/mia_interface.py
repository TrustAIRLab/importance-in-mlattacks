import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from utils import get_model
import argparse
import os
import pickle
import pandas as pd
from metric_base import black_box_benchmarks


class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""
    def __init__(self,root, download, transform,indices,train=True):
        super().__init__(root, train, download, transform)
        self.indices = indices
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target#, index    
    
def setup():
    parser = argparse.ArgumentParser(description='Metric-based Membership Inference Attack')

    parser.add_argument('--gpu_id', type=int, default=1, help='specify which gpu to use')
    parser.add_argument('--iter_num', type=int, default=3, help='specify iteration number')
    parser.add_argument('--divide', type=int, default=2, help='specify which partition')
    parser.add_argument('--random_seed', type=int, default=7, help='fix the random seed for reproduction')
    parser.add_argument('--reduce_type', choices=['large','small','random'], default='large', help='the method of selecting shapley')
    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='cifar10', 
            help='what the training dataset')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
   
    return parser
        
def prepare_candidate_dataset(args):
    shapley_value = np.load(SHAPLEY_PATH)
    value_index = np.argsort(shapley_value)
    large_train_idx = value_index[args.divide*10000:(args.divide+1)*10000]
    if args.dataset == 'cifar10':
        trainset = CIFAR10(root=CIFAR_TRAIN, indices=large_train_idx,
                                                download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)
    return trainloader

def get_blackbox_statistics(dataloader, model):
    """Compute the blackbox statistics (for blackbox attacks)"""
    model.eval()
    crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
    softmax = nn.Softmax(dim=1)

    logits = []
    labels = []
    losses = []
    sample_idx = []
    posteriors = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = crossentropy_noreduce(outputs, targets)
            posterior = softmax(outputs)
            logits.extend(outputs.cpu().numpy())
            posteriors.extend(posterior.cpu().numpy())
            labels.append(targets.cpu().numpy())
            losses.append(loss.cpu().numpy())
    logits = np.vstack(logits)
    posteriors = np.vstack(posteriors)
    labels = np.concatenate(labels)
    losses = np.concatenate(losses)
    return logits, posteriors, losses, labels
        

args = setup().parse_args()
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
args.emb = False

if args.dataset == 'cifar10':
    target_model = get_model(args,10).to(args.device)
target_model.load_state_dict(LOAD_MODEL)

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root=CIFAR_TEST, train=False, download=True, transform=transforms.ToTensor())

if args.dataset == 'cifar10':
    divide_num = 5
    num_classes = 10
        
attack_results = [[] for i in range(4)]
for args.divide in range(divide_num):
    print(f'Divide {args.divide}')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    trainloader = prepare_candidate_dataset(args)
    t_logits_pos, t_posteriors_pos, t_losses_pos, t_labels_pos = get_blackbox_statistics(
            trainloader, target_model)
    t_logits_neg, t_posteriors_neg, t_losses_neg, t_labels_neg = get_blackbox_statistics(
            test_loader, target_model)
    target_train_performance = (t_posteriors_pos, t_labels_pos)
    target_test_performance = (t_posteriors_neg, t_labels_neg)
    mia_bench = black_box_benchmarks(args, target_train_performance, target_test_performance, 
                 target_train_performance, target_test_performance, num_classes)
    corr_acc, conf_acc, entr_acc, modi_acc = mia_bench._mem_inf_auc()
    attack_results[0].append(corr_acc)
    attack_results[1].append(conf_acc)
    attack_results[2].append(entr_acc)
    attack_results[3].append(modi_acc)
print(attack_results)

