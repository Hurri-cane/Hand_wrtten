# author:Hurricane
# date:  2020/11/5
# E-mail:hurri_cane@qq.com
# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
import numpy as np
import struct
import matplotlib.pyplot as plt
import cv2 as cv
import random
import torch
from torch import nn, optim
import torch.nn.functional as F




class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    # num_residuals:残差数
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
def get_net():
    # 构建网络
    # ResNet模型
    model_path = r"F:\PyCharm\Practice\hand_wrtten\logs\Epoch100-Loss0.0000-train_acc1.0000-test_acc0.9930.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))

    net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, 10)))

    # 测试网络
    # X = torch.rand((1, 1, 28, 28))
    # for name, layer in net.named_children():
    #     X = layer(X)
    #     print(name, ' output shape:\t', X.shape)

    # 加载网络模型
    print("Load weight into state dict...")
    stat_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(stat_dict)
    net.to(device)
    net.eval()
    print("Load finish!")
    return net


def predict(img, net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_in = torch.from_numpy(img)
    img_in = torch.unsqueeze(img_in, 0)
    img_in = torch.unsqueeze(img_in, 0).to(device)
    img_in = img_in.float()
    result_org = net(img_in)
    return result_org


