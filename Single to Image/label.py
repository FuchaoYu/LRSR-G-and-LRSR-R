'''
@File  :preprocessing.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 5:07 PM
@Desc  :time-domain raw data to 2-d matrix
'''
import pandas as pd
import scipy.io
import random
import numpy as np
from PIL import Image
import os
import cv2
import csv


def sp(file):
    global lab
    if "B007" in file.split('_', 2)[0]:
        lab = 0
    elif "B014" in file.split('_', 2)[0]:
        lab = 1
    elif "B021" in file.split('_', 2)[0]:
        lab = 2
    elif "IR007" in file.split('_', 2)[0]:
        lab = 3
    elif "IR014" in file.split('_', 2)[0]:
        lab = 4
    elif "IR021" in file.split('_', 2)[0]:
        lab = 5
    elif "OR007" in file.split('_', 2)[0]:
        lab = 6
    elif "OR014" in file.split('_', 2)[0]:
        lab = 7
    elif "OR021" in file.split('_', 2)[0]:
        lab = 8
    elif "NORMAL" in file.split('_', 2)[0]:
        lab = 9

    return lab


def label(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        if "train" in file:
            file_path = folder_path + "/" + file  # 图片文件夹地址
            img_path = os.listdir(file_path)  # 图片地址
            for i in img_path:
                lab = sp(i)
                lab = str(lab)
                path = i.split('.png')[0]
                excl = [path, lab]
                with open('label.csv', 'a', encoding='utf8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(excl)
        # elif ("train") in file:
        #     choices = [i for i in range(data_DE.shape[0] - img_size)]
        #     print(os.path.splitext(file)[0], "val")
        #     choice = random.sample(choices, val_num)
        #     for i in choice:
        #         begin_number = i
        #         img = data_DE[begin_number:begin_number + img_size]
        #         img = np.round((img - min(img)) / (max(img) - min(img)) * 255)
        #         img = np.reshape(img, (int(img_size ** 0.5), int(img_size ** 0.5)))
        #         cv2.imwrite(save_path + '/val/' + os.path.splitext(file)[0] + '_' + str(i) + '.png', img)
        #
        # elif ("val") in file:
        #     choices = [i for i in range(data_DE.shape[0] - img_size)]
        #     print(os.path.splitext(file)[0], "test")
        #     choice = random.sample(choices, test_num)
        #     for i in choice:
        #         begin_number = i
        #         img = data_DE[begin_number:begin_number + img_size]
        #         img = np.round((img - min(img)) / (max(img) - min(img)) * 255)
        #         img = np.reshape(img, (int(img_size ** 0.5), int(img_size ** 0.5)))
        #         cv2.imwrite(save_path + '/test/' + os.path.splitext(file)[0] + '_' + str(i) + '.png', img)
        #
        # else:
        #     print("wrong!", os.path.splitext(file)[0])

        print('Finished creating label ' + file)


data_path = '../../data/12kDriveEnd_img'

label(data_path)
