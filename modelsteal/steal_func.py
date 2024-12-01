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
    
def setup():
    parser = argparse.ArgumentParser(description='Model stealing')

    parser.add_argument('--gpu_id', type=int, default=1, help='specify which gpu to use')
    parser.add_argument('--iter_num', type=int, default=5, help='specify iteration number')
    parser.add_argument('--random_seed', type=int, default=7, help='fix the random seed for reproduction')
    parser.add_argument('--reduce_type', choices=['large','small','random'], default='large', help='the method of selecting shapley')

    parser.add_argument('--query_budget', type=int, default=10, help='specify the size of the trigger')
    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='cifar10', 
            help='what the training dataset')
    parser.add_argument('--stealset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='cifar10', 
            help='what the stealing dataset')
    parser.add_argument('--loss', choices=['l1','l2'], default='l1', 
            help='what loss function is used to steal the model')
    parser.add_argument('--target_type', choices=['full','disjoint'], default='full', 
            help='what reconstruction task')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
   
    return parser

def train(target_model, surrogate_model, optimizer, criterion, trainloader, device):
    surrogate_model.train()
    for i, (images,_) in enumerate(trainloader):
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            logits = target_model(images)
        predict = surrogate_model(images)
        loss = criterion(predict,logits)
        loss.backward()
        optimizer.step()
        
def test(surrogate_model,criterion, cleanloader,device):
    surrogate_model.eval()
    clean_correct = 0
    clean_total = 0
    clean_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cleanloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = surrogate_model(inputs)
            clean_loss = criterion(outputs, targets)

            clean_loss += clean_loss.item()
            _, predicted = outputs.max(1)
            clean_total += targets.size(0)
            clean_correct += predicted.eq(targets).sum().item()

    clean_acc = 100.*clean_correct/clean_total
    print('Clean Accuracy:',clean_acc,'%')
    return clean_acc

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
 
def prepare_surrogate_dataset(args,num_added):
    shapley_value = np.load(SHAPLEY_PATH)
    value_index = np.argsort(shapley_value)
    large_train_idx = value_index[-num_added:]
    small_train_idx = value_index[:num_added]
    print(f'len of the large trainset:{len(large_train_idx)}')
    print(f'len of the small trainset:{len(small_train_idx)}')
    if args.reduce_type == 'large':
        if args.stealset == 'cifar10':
            trainset = CIFAR10(root=CIFAR_TRAIN, indices=large_train_idx,
                                                    download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    elif args.reduce_type == 'small':
        if args.stealset == 'cifar10': 
            randomset = CIFAR10(root=CIFAR_TRAIN, indices=small_train_idx,
                                                    download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(randomset, batch_size=128, shuffle=True, num_workers=0)
    return trainloader

def steal_model(args, target_model, surrogate_model, trainloader, cleanloader, epoch):
    target_model = target_model.to(args.device)
    surrogate_model = surrogate_model.to(args.device)
    print('Evaluating The Target Model')
    clean_acc = test(target_model,nn.CrossEntropyLoss(),cleanloader,args.device)

    if args.loss == 'l2':
        criterion = nn.MSELoss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    optimizer = optim.SGD(surrogate_model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75,90],gamma=0.1)
    for epoch in range(epoch):
        train(target_model, surrogate_model, optimizer, criterion, trainloader, args.device)
        scheduler.step()
    print('Evaluating The Surrogate Model')
    clean_acc = test(surrogate_model,nn.CrossEntropyLoss(),cleanloader,args.device)
    return clean_acc
        

args = setup().parse_args()
print(args)
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
args.emb = False


if args.dataset == 'cifar10':
    target_model = get_model(args,10).to(args.device)
target_model.load_state_dict(TARGET_MODEL)

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root=CIFAR_TEST, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

for iter_num in range(args.iter_num):
    torch.manual_seed(iter_num*17)
    clean_acc_list = []
    for num_added in [100,150,200,300,400,500,700,1000,1500,2000,3000,5000,10000]:
        trainloader = prepare_surrogate_dataset(args,num_added)
        if args.dataset == 'cifar10':
            surrogate_model = get_model(args,10).to(args.device)
        
        surrogate_model.apply(weights_init) 
        args.save_path = SAVE_PATH
        clean_acc = steal_model(args, target_model, surrogate_model, trainloader,test_loader, epoch=100)
        clean_acc_list.append(clean_acc)
        print(clean_acc_list,flush=True)
    with open(ACC_PATH,'wb') as fp:
        pickle.dump(clean_acc_list, fp)