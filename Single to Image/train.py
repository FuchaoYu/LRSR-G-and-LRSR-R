'''
@File  :train.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 6:23 PM
@Desc  :train eval_model [Batch size, channel, width, height]
'''

from model import Model
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from create_dataset import MyDataset
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.optim.lr_scheduler import StepLR
import os
import shutil


def main():
    if os.path.exists('./loss_log'):
        shutil.rmtree('./loss_log')
    writer = SummaryWriter('./loss_log')
    learning_rate = 0.001
    batch_size = 128
    epochs = 200

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg_model_1024 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    vgg_model_1024.features[0] = nn.Conv2d(1, 64, 5)
    vgg_model_1024.classifier[6] = nn.Linear(4096, 786)
    vgg_model_1024.classifier.add_module('add_liner', nn.Linear(786,10))
    model = vgg_model_1024.to(device)
    # for m in eval_model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)

    train_data = MyDataset('../../data/12kDriveEnd_img/train/label.csv', '../../data/12kDriveEnd_img/train')
    # eval_data = MyDataset('../../data/12kDriveEnd_img/val/label.csv', '../../data/12kDriveEnd_img/val')
    test_data = MyDataset('../../data/12kDriveEnd_img/test/label.csv', '../../data/12kDriveEnd_img/test')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(base_optimizer, T_max=40, eta_min=5 * 1e-6)
    scheduler_1 = ExponentialLR(base_optimizer, gamma=0.95)
    scheduler_2 = StepLR(base_optimizer, step_size=3, gamma=0.9)

    def model_test(epoch, dataloader, test_model, highest_score):
        size = len(dataloader.dataset)
        test_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.float().to(device), y.to(device)
                pred = test_model(X)
                # pred = pred.unsqueeze(0)
                # y = y.unsqueeze(0)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.int).sum().item()
        test_loss /= size
        correct /= size
        highest_score = max(highest_score, correct)
        print(f"\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        niter = epoch * len(train_dataloader)
        writer.add_scalars('Train',
                           {'val_correct': correct}, niter)
        return highest_score

    def train_loop(epoch, dataloader, train_model, loss_function, optimizer):
        print('learning_rate: ', optimizer.param_groups[0]['lr'])
        size = len(dataloader.dataset)
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = x.float().to(device)
            pred = train_model(X)
            # pred = pred.unsqueeze(0)
            y = y.to(device)
            # y = y.unsqueeze(0)
            loss = loss_function(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 5 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                niter = epoch * len(train_dataloader) + batch
                writer.add_scalars('Train',
                                   {'train_loss': loss}, niter)
                writer.add_scalars('Train',
                                   {'learning_rate': optimizer.param_groups[0]['lr']}, niter)

    eval_best_correct = 0
    eval_before_correct = 0
    test_best_correct = 0
    test_before_correct = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(t, train_dataloader, model, loss_fn, base_optimizer)
        # eval_best_correct = model_test(t, eval_dataloader, model, eval_best_correct)
        test_best_correct = model_test(t, test_dataloader, model, test_best_correct)
        scheduler_1.step()
        # if eval_best_correct > 0.70 and eval_best_correct > eval_before_correct:
        #     torch.save(model, './eval_model/model_'+str(eval_best_correct)+'.pth')
        #     print('save_model: ', eval_best_correct)

        if test_best_correct > 0.99 and test_best_correct >= test_before_correct:
            torch.save(model, './test_model/model_'+str(test_best_correct)+'.pth')
            print('save_model: ', test_best_correct)
        eval_before_correct = eval_best_correct
        test_before_correct = test_best_correct
        # print('done, eval_best_score: ', eval_best_correct)
        print('done, test_best_score: ', test_best_correct)


if __name__ == '__main__':
    main()
