import torch
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_dataloader(dataset, batch_size, num_workers, transform = None):

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
        
    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                    download=True, transform=transform)

    if dataset == 'imagenet':
        trainset = torchvision.datasets.ImageNet(root='./data/imagenet', split = 'train',
                                                    transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
    
    return trainloader


def get_val_dataloader(dataset, batch_size, num_workers, transform = None, validation_split = 1.0):

    if dataset == 'cifar10':
        valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    if dataset == 'cifar100':
        valset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform)

    if dataset == 'imagenet':
        valset = torchvision.datasets.ImageNet(root='./data/imagenet', split = 'val', transform=transform)

    dataset_size = len(valset)
    indices = list(range(dataset_size))
    v_split = int(np.floor(validation_split * dataset_size))
    val_indices = indices[:v_split]
    val_sampler = SubsetRandomSampler(val_indices)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,sampler=val_sampler,
                                                shuffle=False, num_workers=num_workers)

    return valloader    