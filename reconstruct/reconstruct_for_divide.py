import sys
sys.path.append('..')

from inversion_attack import OptimizationBase
from revealer_attack import Revealer
from utils import train_model,get_model
import argparse
import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import torch.nn as nn
from PIL import Image


def options():
    parser = argparse.ArgumentParser(description='Reconstruct training sample')

    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='celeba', 
            help='what reconstruction task')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
    parser.add_argument('--attack_type', choices=['deepinversion','revealer'], default='revealer', 
            help='what reconstruction attack')
   
    return parser

args = options().parse_args()
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
args.emb = False

if args.dataset == 'cifar10':
    args.num_class = 10
    args.input_shape = [3,32,32] 
    divide_num = 5

for divide in range(divide_num):
    if args.dataset == 'cifar10':
        target_model = get_model(args,10)
    args.target_path = f'reconstruct/target_models/{args.dataset}_{args.arch}_divide_{divide}.pth'
    target_model.load_state_dict(torch.load(DIVIDE_PATH)) # where you save the model
    args.result_path =  os.path.join(SAVE_PATH,args.dataset,args.attack_type,f'divide_{divide}')
    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path)

    white_model = target_model
    if args.attack_type == 'deepinversion':
        rec_generator = OptimizationBase(args,white_model)
        rec_generator.generate_batch(10000)

