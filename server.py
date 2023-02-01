import copy
import random

import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
import math
import utils
import models


class Server(object):
    def __init__(self, participants, rounds, learning_rate, momentum, dataset, device):
        self.participants = participants
        self.dataset = dataset
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model = None
        self.participants_num = len(participants)
        self.participants_updates = []
        self.device = device
        self.current_round = 0

        data_train, data_test = utils.get_dataset(dataset)
        test_batch_size = 10000
        self.test_loader = torch.utils.data.DataLoader(data_test, test_batch_size, shuffle = True, num_workers = 0, pin_memory = False)

        if dataset == "mnist":
            self.model = models.MnistNet()
        else:
            print("not implemented")

    # send the global model to participants and local step
    def dispatch_weights(self):
        for par in self.participants:
            par.step(self.model, self.learning_rate, self.device)

    # receive all updates
    # if asynchronous, recive one then update one and fedavg is replace
    def collect_updates(self):
        for par in self.participants:
            self.participants_updates.append(par.update)

    # update
    def update_the_global_model(self):
        updates = copy.deepcopy(self.participants_updates)
        # fedavg
        # global_model_para = utils.update_global_model(self.model, updates, rate = 1)
        global_model_para = utils.fedavg(self.model, updates)
        self.model.load_state_dict(global_model_para)

    # step
    def step(self):
        self.dispatch_weights()
        self.collect_updates()
        self.update_the_global_model()
        self.participants_updates.clear()