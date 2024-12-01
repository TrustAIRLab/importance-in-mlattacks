import torchvision
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from PIL import Image
import random


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self,args,input_shape,dataset_size):
        self.input_shape = input_shape
        self.dataset_size = dataset_size
        self.num_class = args.num_class

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image = torch.rand(3,32,32).to(self.device)
        label = random.randint(0,self.num_class-1)
        return image

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

        return img, target, idx

def get_dataset(args):
    if args.dataset == 'cifar10':
        return CIFAR10,10, 60000
        