from datetime import datetime
from logging import critical

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

#from BagData import test_dataloader, train_dataloader
from CardiacData import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from Dice_Loss import SoftDiceLoss

def test(epo_num=50, show_vgg_params=False):
    #模型加载与Visdom使能
    vis = visdom.Visdom()#调用Visdom前需先行-python -m visdom.server
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#CUDA可选，无则为CPU
    fcn_model=torch.load('checkpoints_cardiac/fcn_model_25.pt')
    fcn_model = fcn_model.to(device)

    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)#优化器
    #criterion=nn.BCELoss().to(device)
    criterion=SoftDiceLoss().to(device)

    all_test_iter_loss=[]
    for epo in range(epo_num):
        print("This is {} epoch".format(epo))
        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():#强制之后的内容不进行计算图构建
            for index, (bag, bag_msk) in enumerate(test_dataloader):#从测试数据迭代器中取数据
                bag = bag.to(device)
                bag_msk = bag_msk.to(device)#将数据加载到指定设备

                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
                bag_msk_np = np.argmin(bag_msk_np, axis=1)
        
                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                    vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))


if __name__=="__main__":
    test(epo_num=10,show_vgg_params=False)
    
    