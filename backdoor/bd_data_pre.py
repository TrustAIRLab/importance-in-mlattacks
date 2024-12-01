import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from PIL import Image
import copy


transform_valid = transforms.Compose([
        transforms.ToTensor(),
        ])

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

class CIFAR10_BD(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""
    def __init__(self,root, poison_indices, transform,trigger_size, train=True,download=False):
        super().__init__(root, train, download, transform)
        self.poison_indices = poison_indices
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.trigger_size = trigger_size
        self.preprocess_dataset()
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def preprocess_dataset(self, target_label=0):
        '''
        set the target label to 0
        '''
        for idx in self.poison_indices:
            self.targets[idx] = target_label
            self.data[idx] = self.attach_trigger(self.data[idx])
    
    def attach_trigger(self, img):
        bd_img = copy.deepcopy(img)
        width, height, _ = bd_img.shape
        bd_img[width-self.trigger_size-1 : width-1,height-self.trigger_size-1 : height-1,:] = 0
        return bd_img

def get_dataset(args,value_index):
    if args.reduce_type == 'large':
        poison_idx = value_index[-args.poison_num:]
    elif args.reduce_type == 'small':
        poison_idx = value_index[:args.poison_num]
    print(poison_idx)
    if args.dataset == 'cifar10':
        print('cifar10')
        backdoor_training = CIFAR10_BD(root=CIFAR_TRAIN,poison_indices=poison_idx,transform=transform_train, trigger_size=args.poison_size)
        backdoor_testing = CIFAR10_BD(root=CIFAR_TEST,poison_indices=range(10000),transform=transform_valid, trigger_size=args.poison_size,train=False)
        clean_testing = torchvision.datasets.CIFAR10(root=CIFAR_TEST, train=False, download=True, transform=transform_valid)

    backdoor_trainloader = torch.utils.data.DataLoader(backdoor_training, batch_size=128, shuffle=True, num_workers=0)
    backdoor_testloader = torch.utils.data.DataLoader(backdoor_testing, batch_size=128, shuffle=False, num_workers=0)
    clean_testloader = torch.utils.data.DataLoader(clean_testing, batch_size=128, shuffle=False, num_workers=0)
    return backdoor_trainloader, clean_testloader, backdoor_testloader