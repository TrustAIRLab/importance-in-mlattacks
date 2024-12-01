import sys
sys.path.append('..')
from models import *


def get_model(args,num_class):
    if args.dataset == 'cifar10':
        if args.arch == 'resnet18':
            return ResNet18(num_classes=num_class)
        elif args.arch == 'resnet34':
            return ResNet34(num_classes=num_class)
        elif args.arch == 'resnet50':
            return ResNet50(num_classes=num_class)
        elif args.arch == 'mobilenetv2':
            return MobileNetV2(num_classes=num_class)
        else:
            print('We don not support this model yet.')