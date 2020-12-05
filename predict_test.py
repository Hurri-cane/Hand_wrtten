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
import d2lzh_pytorch as d2l
import time
from tqdm import tqdm
import os

orig_path = r"F:\PyCharm\Practice\hand_wrtten\test_imgs"
model_path = r"F:\PyCharm\Practice\hand_wrtten\logs\Epoch100-Loss0.0000-train_acc1.0000-test_acc0.9930.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


if __name__ == '__main__':
    # 构建网络
    # ResNet模型
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))

    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(256, 10)))

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

    # 读取图片
    img_list = os.listdir(orig_path)
    for img_name in img_list:
        img_path = os.path.join(orig_path, img_name)
        # print(img_path)
        img_org = 255 - cv.imread(img_path).astype(np.float32)
        img_gray = cv.cvtColor(img_org, cv.COLOR_RGB2GRAY)
        img = torch.from_numpy(img_gray)
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0).to(device)
        result = net(img)
        best_result = result.argmax(dim=1).item()

        # 显示结果
        img_show = cv.resize(img_org, (600, 600))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_show, "The number is:" + str(best_result), (1, 30), font, 1, (0, 0, 255), 2)
        cv.imshow("result", img_show)
        cv.waitKey(0)
        print(best_result)

    #
    # with torch.no_grad():
    #     X = torch.unsqueeze(img, 1)
    #     if isinstance(net, torch.nn.Module):
    #         net.eval()  # 评估模式, 这会关闭dropout
    #         acc_sum += (net(X.to(device)).argmax(dim=1) == label.to(device)).float().sum().cpu().item()
    #         net.train()  # 改回训练模式
    #     else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
    #         if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
    #             # 将is_training设置成False
    #             acc_sum += (net(X, is_training=False).argmax(dim=1) == label).float().sum().item()
    #         else:
    #             acc_sum += (net(X).argmax(dim=1) == label).float().sum().item()
    #     n += label.shape[0]
    # return acc_sum / n
