import sys
sys.path.append('..')

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils import get_model
import argparse
from distance_measure import GetDistancePGD


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
    parser = argparse.ArgumentParser(description='Check distance distribution')

    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--dataset', choices=['cifar10','mnist','celeba','tinyimagenet'], default='cifar10', 
            help='what the training dataset')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50','mobilenetv2'], default='resnet18', 
            help='what is the architecture of the surrogate model')
    parser.add_argument('--save_path', type=str, help='the path to save the distance')
    parser.add_argument('--shapley_path', type=str, help='the path to the shapley value')
   
    return parser


def prepare_candidate_dataset(args):
    shapley_value = np.load(args.shapley_path)
    value_index = np.argsort(shapley_value)
    large_train_idx = value_index
    if args.dataset == 'cifar10':
        trainset = CIFAR10(root=PATH_TO_CIFAR_DATASET, indices=large_train_idx,
                                                download=False, transform=transforms.ToTensor())
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)

    return trainloader


args = setup().parse_args()
args.device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
args.emb = False

if args.dataset == 'cifar10':
    target_model = get_model(args,10).to(args.device)
else:
    raise NotImplementedError

trainloader = prepare_candidate_dataset(args)
distance_measure = GetDistancePGD(target_model, args.device, args)
distance_measure.calculate_distance(trainloader)

