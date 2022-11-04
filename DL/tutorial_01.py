"""this file is the note of [ What is torch.nn really? ], it introduces a simple linear model with only one layer to
classify photos in MNIST dataset;   its reduced version is named [ tutorial_01_reduction.py ];
    Problem：multi-classification；
    Network Structure： linear、fully-connected layer（with only one neuron） with softmax activation；
        or [ linear、fully-connected layer（with only one neuron）] - output（softmax） ， regard softmax as part of
        multi-classification process，it can be expressed by cross-entropy loss；
    Loss func：cross_entropy（ log_softmax + negative log-likelihood )；
reference: https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-dataset """
import requests
import torch
from pathlib import Path
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import math
import torch.nn.functional as F  # 重构方法1
from torch import nn  # 重构方法2、3
from torch import optim  # 在重构方法2、3的基础上改进训练过程
from torch.utils.data import TensorDataset, DataLoader  # 在重构方法2、3的基础上改进data分割

####################################
# 下载MNIST数据
####################################
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
# PATH.mkdir(parents=True, exist_ok=True)
# URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"
# if not (PATH / FILENAME).exists():
#         content = requests.get(URL + FILENAME).content
#         (PATH / FILENAME).open("wb").write(content)


####################################
# 数据格式转换
####################################
# This dataset is in numpy array format, and has been stored using pickle, a python-specific format for serializing
# data.
# Each image is 28 x 28, and is being stored as a flattened row of length 784 (=28x28).
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
# # Let’s take a look at one; we need to reshape it to 2d first.
# pyplot.imshow(x_train[1].reshape((28, 28)), cmap="gray")
# pyplot.show()
# print(x_train.shape)  # 查看，输出为：(50000, 784)，表示是np的array
# PyTorch uses torch.tensor, rather than numpy arrays, so we need to convert our data.
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape  # x为样本特征数据（5wx784），y为label（5w）；n表示数据量5w条，c表示列，指代784个特征;
# print(x_train, y_train)   # 查看
# print(x_train.shape)  # 查看，输出：torch.Size([50000, 784])
# print(y_train.shape)  # 查看，输出：torch.Size([50000])
# print(y_train.min(), y_train.max())   # 查看，0-9


####################################
# 初始化：超参、weights
####################################
bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for，一个epoch有 训练样本数/bs 次循环，也就是，总共更新weights参数 epochs x 训练样本数 / bs 次
# We are initializing the weights here with Xavier initialisation (by multiplying with 1/sqrt(n)).
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


####################################
# 模型构造与调用演示
####################################
# although PyTorch provides lots of pre-written loss/activation functions, but you can easily write your own using
# plain python.
def log_softmax(x):  # 激活函数，softmax将多个类别的得分归一化，log_softmax就是再取log
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):  # 神经元计算+激活
    return log_softmax(xb @ weights + bias)  # here,@ stands for the dot product operation


def nll(input, target):  # 损失函数，用negative log-likelihood
    return -input[range(target.shape[0]), target].mean()


loss_func = nll


def accuracy(out, yb):  # 评判指标，accuracy，只在这里和重构方法1中用到了
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


# # 重构方法1：
# # 使用 torch.nn.functional，本质还是函数
# # If you’re using negative log likelihood loss and log softmax activation, then Pytorch provides a single function
# # F.cross_entropy that combines the two.
# # 因为是单层的网络，所以直接把激活+loss结合在了一起，就是交叉熵；或者说，不该把softmax作为激活函数，而看作多分类问题处理流程的一部分
# # torch.nn.CrossEntropyLoss就是两个函数的组合nll_loss(log_softmax(input))。
# loss_func = F.cross_entropy
# def model(xb):
#     return xb @ weights + bias
# def accuracy(out, yb):    # 只在原方法和重构方法1中用到了
#     preds = torch.argmax(out, dim=1)
#     return (preds == yb).float().mean()

# # 重构方法2：
# # 使用 torch.nn.Module，改为模型对象，本质是对象
# # 这里也无需再初始化weights，因为模型被抽象成了一个整体
# loss_func = F.cross_entropy
# class Mnist_Logistic(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))    # randn为正态随机，rand为普通的随机
#         self.bias = nn.Parameter(torch.zeros(10))
#     def forward(self, xb):
#         return xb @ self.weights + self.bias
# model = Mnist_Logistic()

# # 重构方法3：
# # 使用 torch.nn.Linear，在重构方法2上更进一步，将线性计算和初始化进一步封装
# # we will instead use the Pytorch class nn.Linear for a linear layer
# loss_func = F.cross_entropy
# class Mnist_Logistic(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin = nn.Linear(784, 10)
#     def forward(self, xb):
#         return self.lin(xb)
# model = Mnist_Logistic()



# 用第一个batch来进行演示，因为没有进行反向传播，因此不会影响后面的训练过程
# 拿第一个batch的数据来进行第一层神经元的计算的具体过程（这里的手写数字识别也可以看作单层神经元的计算+用softmax之类的来实现多分类）
#   xb为64x784的tensor，代表64个样本，784个特征；
#   weights为784x10，代表10个神经元，每个神经元对应784个连接；
#   输出则为64x10，代表64个样本，每个样本对应一个one hot的预测值preds
# S1: 调用模型
xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions，得到的是64（bs）x10（neurons）的tensor
# preds[0], preds.shape     # 查看，第一个样本的preds
# print(preds[0], preds.shape)      # 查看，第一个样本的preds
# S2: 计算损失和准确度
yb = y_train[0:bs]  # yb（64），为一维tensor，所以yb.shape[0] = 64
print(
    loss_func(preds, yb))  # -preds[range(64), yb].mean()；其中preds[range(yb.shape[0]), yb]相当于preds[:,yb]，yb表示实际该预测最大的那个值
print(accuracy(preds, yb))  # 只在这里和重构方法1种用到了

# # 改为重构方法1：一样的，因为本质还是函数

# # 改为重构方法2、3：
# xb = x_train[0:bs]
# yb = y_train[0:bs]
# print(loss_func(model(xb), yb))


####################################
# 训练过程
####################################
for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):  # 0 ~ 781 (range(782)，共782个batch)
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
# Let’s check the loss and accuracy and compare those to what we got earlier.
print(loss_func(model(xb), yb), accuracy(model(xb), yb))  # 如果是重构方法2、3，则需要改为：print(loss_func(model(xb), yb))

# # 改为重构方法1：一样的，因为本质还是model函数

# # 改为重构方法2、3：主要是weights被封装起来了，直接调用对象的方法接口就好了
# # 这里用到的torch.nn.Module接口有：model.parameters()获取参数、 model.zero_grad()使得grad清零
# def fit():
#     for epoch in range(epochs):
#         for i in range((n - 1) // bs + 1):
#             start_i = i * bs
#             end_i = start_i + bs
#             xb = x_train[start_i:end_i]
#             yb = y_train[start_i:end_i]
#             pred = model(xb)
#             loss = loss_func(pred, yb)
#
#             loss.backward()
#             with torch.no_grad():
#                 for p in model.parameters():
#                     p -= p.grad * lr
#                 model.zero_grad()
# fit()

# # 在重构方法2、3的fit基础上引入optim
# # 用到了optim.SGD(parameters, lr)，对梯度下降的循环过程的进一步封装
# opt = optim.SGD(model.parameters(), lr=lr)
# for epoch in range(epochs):
#     for i in range((n - 1) // bs + 1):
#         start_i = i * bs
#         end_i = start_i + bs
#         xb = x_train[start_i:end_i]
#         yb = y_train[start_i:end_i]
#         pred = model(xb)
#         loss = loss_func(pred, yb)
#
#         loss.backward()
#         opt.step()  #这里的step相当于循环
#         opt.zero_grad()

# # 在2、3 fit + optim 的基础上引入dataset和dataloader
# # 用dataset将feature和label对齐；用dataloader自动将数据分为batches并可以依次调用，类似于range
# # dataset使得start_i、end_i变为xb, yb = train_ds[i * bs: i * bs + bs];
# # dataloader进一步使得xb和yb自动分割
# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size=bs)
# opt = optim.SGD(model.parameters(), lr=lr)
# for epoch in range(epochs):
#     for i in train_dl:
#         pred = model(xb)
#         loss = loss_func(pred, yb)
#
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

# # 在2、3 fit + optim + dataset 的基础上，进行validation分割
# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size=bs)
# valid_ds = TensorDataset(x_valid, y_valid)
# valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
# opt = optim.SGD(model.parameters(), lr=lr)
# for epoch in range(epochs):
#     model.train()  # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
#     for xb, yb in train_dl:
#         pred = model(xb)
#         loss = loss_func(pred, yb)
#
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#
#     model.eval()    # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
#     with torch.no_grad():
#         valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
#     print(epoch, valid_loss / len(valid_dl))


# # 训练的核心可以分为三个blocks：数据划分、向前传播（计算损失）、优化参数（loss反向回传、梯度下降）；
# # 在含validation的训练中，顺序为：[数据划分]、[向前传播（train set）、优化参数、向前传播（validation set）]
# # 将三个blocks分别封装为：数据划分-get_data、向前传播-loss_batch、优化参数-opt；在此基础上进一步封装[向前传播、优化参数、向前传播]过程
# def get_data(train_ds, valid_ds, bs):
#     return (
#         DataLoader(train_ds, batch_size=bs, shuffle=True),
#         DataLoader(valid_ds, batch_size=bs * 2),
#     )
#
# def get_model():
#     model = Mnist_Logistic()
#     return model, optim.SGD(model.parameters(), lr=lr)
#
# def loss_batch(model, loss_func, xb, yb, opt=None):
#     loss = loss_func(model(xb), yb)
#     if opt is not None:
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#     return loss.item(), len(xb)
#
# def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
#     for epoch in range(epochs):
#         model.train()   # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
#         for xb, yb in train_dl:
#             loss_batch(model, loss_func, xb, yb, opt)
#         model.eval()    # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
#         with torch.no_grad():
#             losses, nums = zip(
#                 *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
#             ) # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式
#         val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
#         print(epoch, val_loss)
#
# # 封装之后，整个训练过程变为：
# train_ds = TensorDataset(x_train, y_train)
# valid_ds = TensorDataset(x_valid, y_valid)
# train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# model, opt = get_model()
# fit(epochs, model, loss_func, opt, train_dl, valid_dl)
