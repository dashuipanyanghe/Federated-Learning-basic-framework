from collections import OrderedDict
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import datasets, transforms
import numpy as np


def get_dataset(dataset):
    dataset_train = 0
    dataset_test = 0
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('../DATA', train=True,
                                  transform=transform)
        # print(type(dataset_train))
        dataset_test = datasets.MNIST('../DATA', train=False,
                                  transform=transform)
    else:
        print("not implemented")
    return dataset_train, dataset_test

# trans the para dict to list
def deal_update(update):
    param_update = []
    for key in update.keys():
        param_update = update[key].view(-1) if not len(param_update) else torch.cat((param_update, update[key].view(-1)))
    return param_update

# recover the dict
def recover_the_update(global_model, list_update):
    update = OrderedDict()
    start_idx = 0
    for key in global_model.state_dict().keys():
        param = list_update[start_idx:start_idx+len(global_model.state_dict()[key].view(-1))].reshape(global_model.state_dict()[key].shape)
        # print(param.equal(global_model.state_dict()[key]))
        # param.cuda()
        key_value = OrderedDict({key:param})
        update.update(key_value)
        start_idx = start_idx + len(global_model.state_dict()[key].view(-1))
    return update


# fedavg
def fedavg(global_model, updates):
    temp = sum(updates)/len(updates)
    temp_final_update = recover_the_update(global_model, temp)
    global_model.load_state_dict(temp_final_update)
    return global_model.state_dict()

# test
def test(model, device, test_loader, print_acc = True):
    model.eval()
    test_loss = 0
    correct = 0
    # print("i am testing")
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction = 'sum').item()
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    if print_acc == True:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    return acc, test_loss