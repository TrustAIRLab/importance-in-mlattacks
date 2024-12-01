import sys
sys.path.append('..')

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils import get_model
import argparse
from loss_measure import GetLoss


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

        return img, target, index
    
def setup():
    parser = argparse.ArgumentParser(description='CHeck loss distribution')

    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--iter_num', type=int, default=3, help='specify iteration number')
    parser.add_argument('--random_seed', type=int, default=7, help='fix the random seed for reproduction')
    parser.add_argument('--reduce_type', choices=['large','small','random'], default='large', help='the method of selecting shapley')

    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='cifar10', 
            help='what the training dataset')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
   
    return parser

def prepare_candidate_dataset(args):
    shapley_value = SHAPLEY_PATH
    value_index = np.argsort(shapley_value)
    large_train_idx = value_index
    print(f'len of the large trainset:{len(large_train_idx)}')
    if args.dataset == 'cifar10':
        trainset = CIFAR10(root=CIFAR_TRAIN, indices=large_train_idx,
                                                download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)

    return trainloader

args = setup().parse_args()
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
args.emb = False

if args.dataset == 'cifar10':
    target_model = get_model(args,10).to(args.device)
target_model.load_state_dict(LOAD_PATH)

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root=CIFAR_TEST, train=False, download=True, transform=transforms.ToTensor())

trainloader = prepare_candidate_dataset(args)
loss_measure = GetLoss(target_model, args.device, args)
loss_measure.calculate_loss(trainloader)

