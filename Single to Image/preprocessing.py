'''
@File  :preprocessing.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 5:07 PM
@Desc  :time-domain raw data to 2-d matrix
划分思想：负载0和3作为训练集，1作为验证机，2作为测试集
'''

import scipy.io
import random
import numpy as np
import os
import cv2


def folder2img(save_path, folder_path, train_num, val_num, test_num, img_size):
    files = os.listdir(folder_path)   # 遍历12kDriveEnd中的文件
    if not os.path.exists(save_path + '/train'):   # 在save_path中创建文件夹
        os.makedirs(save_path + '/train')
    # if not os.path.exists(save_path + '/val'):
    #     os.makedirs(save_path + '/val')
    # if not os.path.exists(save_path + '/test'):
    #     os.makedirs(save_path + '/test')
    for file in files:   # 在原始数据集(12kDriveEnd)上操作
        #        print(type(os.path.splitext(file)[1]))
        if os.path.splitext(file)[1] == '.mat':  # 找出其中.mat文件 （分割文件为列表[0,1,2,...]，以.为界限）
            file_path = folder_path + "/" + file  # mat文件地址
            raw_data = scipy.io.loadmat(file_path)  # 打开mat文件
            for name in raw_data.keys():
                if "DE" in name:       # 将DE信号作为目标信号
                    column_DE = name   # column_DE = X118_DE_time
                    print(column_DE)
            data_DE = raw_data[column_DE]
            print(data_DE.shape[0])
            print(min(data_DE))
            # if ("_3" in os.path.splitext(file)[0]) or ("_0" in os.path.splitext(file)[0]):  # 0和3作为训练集 # 25*8000个图像
            choices = [i for i in range(data_DE.shape[0] - img_size)]    # 保证选取的数据点在最后一位的64*64个前
            print(os.path.splitext(file)[0], "train")
            choice = random.sample(choices ,train_num)    # 随机选取8000个数据点
            for i in choice:
                begin_number = i
                img = data_DE[begin_number:begin_number + img_size]   # 选取图像大小的数据 可以生成8000个图像
                img = np.round((img - min(img)) / (max(img) - min(img)) * 255)
                img = np.reshape(img, (int(img_size ** 0.5), int(img_size ** 0.5)))  # 将一维数组变为二维图像(reshap转化)
                cv2.imwrite(save_path + '/train/' + os.path.splitext(file)[0] + '_' + str(i) + '.png', img)

            # elif ("_1") in os.path.splitext(file)[0]:
            #     choices = [i for i in range(data_DE.shape[0] - img_size)]
            #     print(os.path.splitext(file)[0], "val")
            #     choice = random.sample(choices, val_num)
            #     for i in choice:
            #
            #         begin_number = i
            #         img = data_DE[begin_number:begin_number + img_size]
            #         img = np.round((img - min(img)) / (max(img) - min(img)) * 255)    # 0-1的像素值
            #         img = np.reshape(img, (int(img_size ** 0.5), int(img_size ** 0.5)))
            #         cv2.imwrite(save_path + '/val/' + os.path.splitext(file)[0] + '_' + str(i) + '.png', img)
            #
            # elif ("_2") in os.path.splitext(file)[0]:
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

            print('Finished creating img files for ' + file)


data_path = '../../data/12kDriveEnd'
normal_path = '../../data/Normal_Baseline_Data'
img_save_path = '../../data/12kDriveEnd_img'
train_number = 1000  # 8000 每一类（负载、大小、位置）故障下的数据数量
val_number = 100  # 1000   每一类（负载、大小、位置）故障下的数据数量
test_number = 100 # 1000   每一类（负载、大小、位置）故障下的数据数量
matrix_size = 128 * 128   #64*64
# 设想：每一类取100个样本，则每组位置下1300个样本
folder2img(img_save_path, data_path, train_number, val_number, test_number, matrix_size)

# signal2image(data_path, matrix_number, matrix_size)
