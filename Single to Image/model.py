'''
@File  :eval_model.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 5:06 PM
@Desc  :LeNet5 Pytorch_ver
'''

from torch.nn import Module
from torch import nn



class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))  # 64*64*32
        self.relu1 = nn.ReLU()  # LeNet5 doesn't use ReLU
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 32*32*32  LeNet5 uses average pool

        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))  # 32*32*64
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 16*16*64

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))  # 16*16*128
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 8*8*128

        self.conv4 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))  # 8*8*256
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 4*4*256

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4096, 2560)
        self.relu5 = nn.ReLU()  # LeNet5 uses tanh
        self.fc2 = nn.Linear(2560, 768)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(768, 10)
        self.relu7 = nn.ReLU()

        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = self.pool3(y)
        y = self.conv4(y)
        y = self.relu4(y)
        y = self.pool4(y)

        y = self.flatten(y)
        y = y.view(-1, 4*4*256)
        y = self.fc1(y)
        y = self.relu5(y)
        y = self.fc2(y)
        y = self.relu6(y)
        y = self.fc3(y)
        y = self.relu7(y)

        #y = self.softmax(y)
        return y


