"""this file is the note of [ What is torch.nn really? ], it introduces a 3-layer CNN model to classify photos in MNIST
dataset;    it is based on the analysis of [ tutorial_01.py ];
    Problem：multi-classification；
    Network Structure： [ conv（ activation： relu ） - conv（relu） - conv（relu） - avg_pooling ] - output（softmax）， it
        regards softmax as part of multi-classification process，which can be expressed by cross-entropy loss, rather
        than activation function；
    Loss func：cross_entropy；
reference: https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-dataset """
import requests
import torch
from pathlib import Path
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import math
import torch.nn.functional as F
from torch import nn
from torch import optim  # 在重构方法2、3的基础上改进训练过程
from torch.utils.data import TensorDataset, DataLoader  # 在重构方法2、3的基础上改进data分割


####################################
# 导入MNIST数据 + 数据格式转换
####################################
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
# 从pickle（a python-specific format for serializing data）到numpy的row  （类似于解压？？？）
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)  # 从numpy到tensor
n, c = x_train.shape

####################################
# 初始化：超参
####################################
bs = 64  # batch size
lr = 0.1  # learning rate
epochs = 2  # how many epochs to train for


####################################
# 模型构造、函数封装
####################################
class Mnist_CNN(nn.Module):  # 主要声明forward网络结构、__init__初始化
    def __init__(self):
        super().__init__()  # 参数初始化，因为activation是relu，pooling也不需要参数，所以只需要初始化卷积层
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
        # bias=True)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # 输入图片为 1channel x 28pixels x 28pixels ；第一层filter为16个，大小为 1channel x 3 x 3；第一层输出为 16 channels x 14 x 14；
        # 问题：为什么这里的filter个数定为16？
        # 分析：filter（卷积核）的大小一般为奇数x奇数；filter个数一般设为2的指数，即2、4、8、16 etc
        # 问题：为什么这里的stride定为2？
        # 分析：28 + 2（padding） = 30 ； 30 - 3 （kernel_size） = 27 ； 27 / 2（stride） = 13.5，向下取整为13； 13+1 = 14；
        # 这里实际上不适合用stride = 2，因为 ( input_size + padding x 2 - kernel_size ) / （stride）不为整数；
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        # 第二层filter为16个，大小为 16channels x 3 x 3；第二层输出为 16 channels x 7 x 7；
        # ( input_size + padding x 2 - kernel_size ) / （stride）= （14 + 2 - 3）/2 = 6.5 → 6 ；  6 + 1 = 7；
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        # 第三层filter为10个，大小为 16channels x 3 x 3；第三层输出为 10 channels x 4 x 4 ;

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)  # N张图(batch_size) x 1channel x 28pixels x 28pixels
        xb = F.relu(self.conv1(xb))  # 网络结构为：conv（relu） - conv（relu） - conv（relu） - avg_pooling
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        # 如果用了padding，深度（channels）靠Conv，尺度靠pooling
        # CLASS torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        # 第三层输出为 10 channels x 4 x 4 ； kernal_size为 10 x 4 x 4 ，tride为 None; 则最终输出为 10 channels x 1 x 1 ；
        return xb.view(-1, xb.size(1))
        # 池化后输出为 N张图(batch_size) x 10 channels x 1 x 1 ; xb.size(1) = 10 ; 这里转化为 N张图(batch_size) x 10;


def get_model():
    model = Mnist_CNN()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # 采用带momentum的SGD方法
    return model, opt


# # 改写1：用nn.Sequential改写model
# # 将class Mnist_CNN(nn.Module)改到get_model中
# # 用Lambda的原因在于：nn.Sequential只接受 a Module subclass，而x.view()为一个tensor，需要有一个类（Lambda）进行转换
# # 也不是一定要用preprocess！！！get_model中line 104 Lambda(preprocess)直接写作Lambda(lambda x: x.view(-1, 1, 28, 28))也可以
# class Lambda(nn.Module):    # 作用于模型构造，因为nn.Sequential只接受 a Module subclass；Lambda的接收对象为
#     def __init__(self, func):
#         super().__init__()
#         self.func = func
#
#     def forward(self, x):
#         return self.func(x)
#
#
# def preprocess(x):    # 在model内部使用！！！model只处理 dataloader 对象的features，也就是只有x；也不一定要用该方法，见line 88说明
#     return x.view(-1, 1, 28, 28)
#
#
# def get_model():
#     model = nn.Sequential(
#         Lambda(preprocess),
#         # 将原始2维数据转换为4维；nn.Sequential只接受 a Module subclass，而x.view()为一个tensor，需要有一个类（Lambda）进行转换
#         nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(),
#         nn.AvgPool2d(4),  # 本质是 nn.functional.avg_pool2d（）和 nn.AvgPool2d（） 的区别
#         Lambda(lambda x: x.view(x.size(0), -1)),  # 将4维数据转换为原始2维( N张图(batch_size) x new features )
#     )     # 为什么用lambda x: x.view()？ 因为nn.Sequential自动传输每一层的输入输出，没有显性的可以直接调用的x，只有用lambda
#     opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     return model, opt


# # 改写2：在1的基础上修改图片的尺度：拓展适应性
# class Lambda(nn.Module):    # 作用于模型构造，因为nn.Sequential只接受 a Module subclass
#     def __init__(self, func):
#         super().__init__()
#         self.func = func
#
#     def forward(self, x):
#         return self.func(x)
#
#
# def preprocess(x, y):     # 直接基于 dataloader 对象，是features和labels对齐了的，所以不能只有x； 其更重要的是方便gpu调用
#     return x.view(-1, 1, 28, 28), y
#
#
# class WrappedDataLoader:  # dl = dataloader 直接传数据； func 传入 preprocess 用来转换数据格式； 更重要的是方便gpu调用
#     def __init__(self, dl, func):
#         self.dl = dl
#         self.func = func
#
#     def __len__(self):
#         return len(self.dl)
#
#     def __iter__(self):
#         batches = iter(self.dl)
#         for b in batches:
#             yield (self.func(*b))
#
# def get_model():
#     model = nn.Sequential(    # 1中第一层数据转换变为了WrappedDataLoader中的一部分
#         nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),     #虽然拓展了尺度，但是channels还是没有变
#         nn.ReLU(),
#         nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(),
#         # replace nn.AvgPool2d with nn.AdaptiveAvgPool2d, which allows us to define the size of the output tensor we
#         # want, rather than the input tensor we have.
#         nn.AdaptiveAvgPool2d(1),
#         Lambda(lambda x: x.view(x.size(0), -1)),
#     )     # 为什么用lambda x: x.view()？ 因为nn.Sequential自动传输每一层的输入输出，没有显性的可以直接调用的x，只有用lambda
#     opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     return model, opt


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()  # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()  # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )  # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式(将元组解压为列表，才能应用numpy中的计算)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


# 封装之后，整个训练过程变为：
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
loss_func = F.cross_entropy
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# # 改写1：不变

# # 改写2：拓展图片尺度后，整个训练过程变为：
# train_ds = TensorDataset(x_train, y_train)
# valid_ds = TensorDataset(x_valid, y_valid)
# train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# train_dl = WrappedDataLoader(train_dl, preprocess)
# valid_dl = WrappedDataLoader(valid_dl, preprocess)
# model, opt = get_model()
# loss_func = F.cross_entropy
# fit(epochs, model, loss_func, opt, train_dl, valid_dl)
