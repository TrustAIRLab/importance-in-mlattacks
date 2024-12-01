import sys
sys.path.append('..')

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import argparse
from utils import train_model,get_model
import pandas as pd


transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

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

        return img, target
        
def options():
    parser = argparse.ArgumentParser(description='Reconstruct training sample')

    parser.add_argument('--gpu_id', type=int, default=2, help='specify which gpu to use')
    parser.add_argument('--k', type=int, default=6, help='specify K value')
    parser.add_argument('--random_seed', type=int, default=7, help='fix the random seed for reproduction')
    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='celeba', 
            help='what reconstruction task')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
   
    return parser

args = options().parse_args()
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
shapley_value = np.load(SHAPLEY_PATH)
value_index = np.argsort(shapley_value)

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root=CIFAR_TEST, train=False, download=True, transform=transforms.ToTensor())

print(len(testset))
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

if args.dataset == 'cifar10':
    divide_num = 5
for divide in range(divide_num):
    if args.dataset == 'cifar10':
        target_model = get_model(args,10)
    args.save_path = SAVE_PATH_FOR_DIVIDE
    large_train_idx = value_index[divide*10000:(divide+1)*10000]
    print(f'len of the large trainset:{len(large_train_idx)}')
    if args.dataset == 'cifar10':
        trainset = CIFAR10(root=CIFAR_TRAIN, indices=large_train_idx,
                                                download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    train_model(args, target_model,trainloader,test_loader, epoch=100)