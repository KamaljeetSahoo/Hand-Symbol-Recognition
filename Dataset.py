# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:19:42 2020

@author: Kamaljeet
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


datadir = "mini_alpabet_dataset"

def load_split_train_test(datadir, valid_size = .2, batch_size=4):
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=4)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=4)
    return trainloader, testloader

