'''
@File  :create_dataset.py
@Author:Zhiwei Zheng
@Date  :5/10/2021 1:58 PM
@Desc  :creating a Custom Dataset for my files
'''

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        img_path = img_path+'.png'
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample


# data = MyDataset('../../data/12kDriveEnd_img/train/label.csv', '../../data/12kDriveEnd_img/train/')
# img_data = DataLoader(dataset=data, batch_size=10, shuffle=True)
# for batch, (X, y) in enumerate(img_data):
#     print(y)
#     break


