import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np
import copy
import utils
import models


class Participant(object):
    def __init__(self, participant_id, batch_size, local_epochs, momentum, participant_num, dataset):

        self.id = participant_id
        self.participants_num = participant_num
        self.batch_size = batch_size
        self.momentum = momentum
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.criterion = nn.NLLLoss()
        self.learning_rate = None
        self.update = None
        self.model = None
        self.sampler = None

        if dataset == "mnist":
            self.model = models.MnistNet()
        else:
            print("not implemented")

        # iid data
        data_train, data_test = utils.get_dataset(self.dataset)
        if self.participants_num > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(data_train, num_replicas=self.participants_num,
                                                                           rank=self.id)
            self.train_loader = torch.utils.data.DataLoader(
                data_train, sampler = self.sampler,
                batch_size = self.batch_size, shuffle = self.sampler is None, num_workers=0, pin_memory=True)
        elif self.participants_num == 1:
            self.train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=0, pin_memory=True)
        else:
            print("participants num error")


    # train
    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        print("Here is the Participant {} work! local epoch:{}".format(self.id, epoch))
        # for data, target in train_loader:
        #     print(data)
        #     print(target)
        if (self.dataset == "mnist"):
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = F.nll_loss(outputs, targets)
                loss.backward()
                optimizer.step()

    # step: compute the update
    def step(self, global_model, learning_rate, device):
        self.learning_rate = learning_rate
        self.model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4,
                              momentum=self.momentum)
        epochs = self.local_epochs
        self.model.to(device)
        for iter in range(epochs):
            self.train(self.model, device, self.train_loader, optimizer, iter)
            # self.test(device)
        # update and model
        # self.update = utils.compute_update(self.model.state_dict(), global_model.state_dict())
        self.update = self.model.state_dict()
        self.update = utils.deal_update(self.update)