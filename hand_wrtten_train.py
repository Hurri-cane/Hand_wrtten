
# author:Hurricane
# date:  2020/11/4
# E-mail:hurri_cane@qq.com

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

# 训练集文件
train_images_idx3_ubyte_file = 'F:/PyCharm/Practice/hand_wrtten/dataset/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'F:/PyCharm/Practice/hand_wrtten/dataset/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 'F:/PyCharm/Practice/hand_wrtten/dataset/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'F:/PyCharm/Practice/hand_wrtten/dataset/t10k-labels.idx1-ubyte'


# 读取数据部分
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'  # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量: %d张, 图片大小: %d*%d' % (num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(
        image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, 28, 28))
    # plt.figure()
    for i in tqdm(range(num_images)):
        image = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols)).astype(np.uint8)
        # images[i] = cv.resize(image, (96, 96))
        images[i] = image
        # print(images[i])
        offset += struct.calcsize(fmt_image)

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量: %d张' % (num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in tqdm(range(num_images)):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


# 构建网络部分
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


def evaluate_accuracy(img, label, net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        X = torch.unsqueeze(img, 1)
        if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == label.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
        else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
            if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == label).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == label).float().sum().item()
        n += label.shape[0]
    return acc_sum / n


if __name__ == '__main__':
    print("train:")
    train_images_org = load_train_images().astype(np.float32)
    train_labels_org = load_train_labels().astype(np.int64)
    print("test")
    test_images = load_test_images().astype(np.float32)[0:1000]
    test_labels = load_test_labels().astype(np.int64)[0:1000]
    # 数据转换为Tensor
    train_images = torch.from_numpy(train_images_org)
    train_labels = torch.from_numpy(train_labels_org)
    test_images = torch.from_numpy(test_images)
    test_labels = torch.from_numpy(test_labels)
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(5):
        j = random.randint(0, 60000)
        print("now, show the number of image[{}]:".format(j), int(train_labels_org[j]))
        img = train_images_org[j]
        img = cv.resize(img, (600, 600))
        cv.imshow("image", img)
        cv.waitKey(0)
    cv.destroyAllWindows()
    print('all done!')
    print("*" * 50)

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
    X = torch.rand((1, 1, 28, 28))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, ' output shape:/t', X.shape)

    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr, num_epochs = 0.001, 100
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    batch_size = 1000
    net = net.to(device)

    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    loop_times = round(60000 / batch_size)
    train_acc_plot = []
    test_acc_plot = []
    loss_plot = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        for i in tqdm(range(1, loop_times)):
            x = train_images[(i - 1) * batch_size:i * batch_size]
            y = train_labels[(i - 1) * batch_size:i * batch_size]
            x = torch.unsqueeze(x, 1)  # 对齐维度
            X = x.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_images, test_labels, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        torch.save(net.state_dict(), 'logs/Epoch%d-Loss%.4f-train_acc%.4f-test_acc%.4f.pth' % (
            (epoch + 1), train_l_sum / batch_count, train_acc_sum / n, test_acc))
        print("save successfully")

        test_acc_plot.append(test_acc)
        train_acc_plot.append(train_acc_sum / n)
        loss_plot.append(train_l_sum / batch_count)

    x = range(0,100)
    plt.plot(x,test_acc_plot,'r')
    plt.plot(x, train_acc_plot, 'g')
    plt.plot(x, loss_plot, 'b')
    print("*" * 50)
