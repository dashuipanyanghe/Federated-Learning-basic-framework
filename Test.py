import torch
import time
import gc
import math
import copy
import numpy as np

from server import Server
from participant import Participant
import utils


# example on mnist
def test_mnist():
    np.random.seed(40)
    # the data dir
    # datadir = ""

    # parameter settings
    dataset = "mnist"
    global_rounds = 30
    learning_rate = 0.05
    momentum = 0.9
    # local setting
    local_batch_size = 64
    local_epochs = 1
    local_momentum = 0.9
    participants_num = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # generate participants
    participants = []
    for participant_id in range(participants_num):
        participants.append(Participant(participant_id,local_batch_size,local_epochs,local_momentum,participants_num,dataset))

    # generate server
    server = Server(participants, global_rounds, learning_rate, momentum, dataset, device)

    server.model.to(device)

    # FL
    for i in range(server.rounds):
        print("Training! current round:",i)
        server.step()
        acc,loss = utils.test(server.model,server.device,server.test_loader)


if __name__ == '__main__':
    test_mnist()

