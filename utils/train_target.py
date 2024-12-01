import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from utils import get_dataset
import os


def train(target_model, optimizer, criterion, epoch, args,trainloader):
    print('\nEpoch: %d' % epoch)
    device = args.device
    target_model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = target_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print(f'Train Accuracy:{acc:.3f}%, loss: {train_loss:.3f}',flush=True)

def test(net,criterion,args,testloader):
    device = args.device
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print(f'Test Accuracy:{acc:.3f}%, loss: {test_loss:.3f}',flush=True)
    torch.save(net.state_dict(), os.path.join(SAVE_PATH,args.save_path))
    return acc

def train_model(args,target_model,trainloader,testloader,epoch=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(target_model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    target_model.to(args.device)
    for epoch in range(epoch):
        train(target_model, optimizer, criterion, epoch, args, trainloader)
        if epoch % 10 == 0:
            test(target_model, criterion, args, testloader)
        scheduler.step()
    test(target_model, criterion, args, testloader)
           
def train_backdoor_model(args,target_model,trainloader,testloader,testbackdoor,epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(target_model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    target_model.to(args.device)
    for epoch in range(epochs):
        train(target_model, optimizer, criterion, epoch, args, trainloader)
        if epoch % 10 == 0:
            test(target_model, criterion, args, testloader)
            test(target_model, criterion, args, testbackdoor)
        scheduler.step()
        if epoch == epochs - 1:
            backdoor_acc = test(target_model, criterion, args, testbackdoor)
            clean_acc = test(target_model, criterion, args, testloader)
    print('=====Clean====')
    test(target_model, criterion, args, testloader)
    print('=====Backdoor====')
    test(target_model, criterion, args, testbackdoor)
    return backdoor_acc, clean_acc

def evaluate_target_model(args, target_model):
    device = args.device
    dataset = torchvision.datasets.CIFAR10(root=CIFAR_TEST, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    target_model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = target_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print('Accuracy:',acc,'%')


