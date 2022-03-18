import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])#mean:均值；std：标准差

class CardiacDataset(Dataset):

    def __init__(self, transform=None):#初始化
        self.transform = transform
        
    def __len__(self):#数据集大小
        return len(os.listdir('cardiac_data'))#读取原始心脏数据集

    def __getitem__(self, idx):
        img_name = os.listdir('cardiac_data')[idx]#获取图像文件名
        imgA = cv2.imread('cardiac_data/'+img_name)#读取图像
        imgA = cv2.resize(imgA, (160, 160))#统一图像尺寸至160*160
        imgB = cv2.imread('cardiac_data_msk/'+img_name, 0)#读取标签文件名（同图像）
        imgB = cv2.resize(imgB, (160, 160))#统一尺寸
        imgB = imgB/255#图像二值化
        imgB = imgB.astype('uint8')#转变数据类型
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2,0,1)#图像转置
        imgB = torch.FloatTensor(imgB)#转换至Tensor
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)    

        return imgA, imgB

cardiac = CardiacDataset(transform)

train_size = int(0.9 * len(cardiac))
test_size = len(cardiac) - train_size
train_dataset, test_dataset = random_split(cardiac, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
