import sys
sys.path.append('..')

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from bd_data_pre import get_dataset
from utils import train_backdoor_model, get_model
import argparse
import os
import pickle

    
    
def setup():
    parser = argparse.ArgumentParser(description='Backdoor attack')

    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--iter_num', type=int, default=5, help='specify iteration number')
    parser.add_argument('--random_seed', type=int, default=7, help='fix the random seed for reproduction')
    parser.add_argument('--reduce_type', choices=['large','small','random'], default='large', help='the method of selecting shapley')
    parser.add_argument('--poison_size', type=int, default=2, help='specify the size of the trigger')
    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='cifar10', 
            help='what reconstruction task')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
   
    return parser

args = setup().parse_args()
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
args.emb = False
shapley_value = np.load(SHAPLEY_PATH)
value_index = np.argsort(shapley_value)

print(f'The size of the trigger:{args.poison_size}')

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
    
for iter_num in range(args.iter_num):
    torch.manual_seed(iter_num*17)
    
    backdoor_acc_list = []
    clean_acc_list = []
    for args.poison_num in [30, 50,100,150,200,300,500,700,1000,2000,3000,5000]:
        args.save_path = SAVE_PATH
        training_backdoor, testing_clean, testing_backdoor = get_dataset(args,value_index)
        print(f'The size of the training dataset:{len(training_backdoor.dataset)}')
        print(f'The size of the backdoor test dataset:{len(testing_backdoor.dataset)}')
        print(f'The size of the clean test dataset:{len(testing_clean.dataset)}')
        if args.dataset == 'cifar10':
            target_model = get_model(args,10)
        target_model.apply(weights_init) 
        print('Start training', flush=True)
        bd_acc, clean_acc = train_backdoor_model(args, target_model,training_backdoor,testing_clean,testing_backdoor, epochs=100)
        backdoor_acc_list.append(bd_acc)
        clean_acc_list.append(clean_acc)
        print(f'clean acc: {clean_acc_list}',flush=True)
        print(f'backdoor acc: {backdoor_acc_list}',flush=True)
    with open(ASR_PATH,'wb') as fp:
        pickle.dump(backdoor_acc_list, fp)
    with open(ACC_PATH,'wb') as fp:
        pickle.dump(clean_acc_list, fp)