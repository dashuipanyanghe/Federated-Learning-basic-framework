import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.add_module("conv1",nn.Conv2d(1, 10, kernel_size=5))
        self.add_module("conv2",nn.Conv2d(10, 20, kernel_size=5))
        self.add_module("dropout1",nn.Dropout2d())
        self.add_module("fc1",nn.Linear(320,50))
        self.add_module("fc2",nn.Linear(50,10))
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.dropout1(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        output = F.log_softmax(x,dim=1)
        return output
