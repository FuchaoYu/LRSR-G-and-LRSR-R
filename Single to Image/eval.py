'''
@File  :eval.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 6:24 PM
@Desc  :evaluate eval_model
'''

from model import Model
import torch
from torch import nn
from torch.utils.data import DataLoader
from create_dataset import MyDataset
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.optim.lr_scheduler import StepLR


def model_test(dataloader, test_model):
    size = len(dataloader.dataset)
    test_model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.float().to(device), y.to(device)
            pred = test_model(X)
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
    correct /= size
    return correct


if __name__ == '__main__':
    batch_size = 100
    test_data = MyDataset('../../data/12kDriveEnd_img/test.csv', '../../data/12kDriveEnd_img/test')
    train_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('./eval_model/model_0.7716428571428572.pth')
    precision = model_test(train_dataloader, model)
    print(precision)
