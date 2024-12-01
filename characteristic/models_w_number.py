import sys
sys.path.append('..')

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils import train_model, get_model, test
import argparse
import pandas as pd
import pickle
import torch.nn as nn


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
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        
    
def options():
    parser = argparse.ArgumentParser(description='Explore training difference between high and low importance samples')

    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--random_seed', type=int, default=7, help='fix the random seed for reproduction')
    parser.add_argument('--reduce_type', choices=['KNN-shapley','random','large','small'], default='KNN-shapley', help='the method of selecting shapley')
    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='cifar10', 
            help='what reconstruction task')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
   
    return parser

args = options().parse_args()
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
args.emb = False
shapley_value = np.load(SHAPLEY_PATH)
value_index = np.argsort(shapley_value)

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root=CIFAR_TRAIN, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

for args.reduce_type in ['large','small']:
    for iter_num in range(5):
        torch.manual_seed(iter_num*17)
        acc_list = []
        for num_added in [50, 100, 150, 200, 300, 400, 500, 700, 1000,2000, 5000]:
            args.save_path = SAVE_PATH
            if args.dataset == 'cifar10':
                target_model = get_model(args,10)
               
            target_model.apply(weights_init)  
            large_train_idx = value_index[-num_added:]
            small_train_idx = value_index[:num_added]
            print(f'len of the large trainset:{len(large_train_idx)}')
            print(f'len of the small trainset:{len(small_train_idx)}')
            if args.reduce_type == 'large':
                if args.dataset == 'cifar10':
                    trainset = CIFAR10(root='/u/nkp2mr/rui/data/cifar10', indices=large_train_idx,
                                                            download=False, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
            elif args.reduce_type == 'small':
                if args.dataset == 'cifar10': 
                    randomset = CIFAR10(root='/u/nkp2mr/rui/data/cifar10', indices=small_train_idx,
                                                            download=False, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(randomset, batch_size=128, shuffle=True, num_workers=0)
            train_model(args,target_model,trainloader,test_loader, epoch=100)
            acc_list.append(test(target_model,nn.CrossEntropyLoss(),args,test_loader))
            print(acc_list,flush=True)
        with open(ACC_PATH,'wb') as fp:
            pickle.dump(acc_list, fp)