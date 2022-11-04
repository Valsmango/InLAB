"""this file is the note of [ What is torch.nn really? ], it introduces a simple linear model with only one layer to
classify photos in MNIST dataset;   it is the reduced version of file [ tutorial_01.py ];
    Problem：multi-classification；
    Network Structure： linear、fully-connected layer（with only one neuron） with softmax activation；
        or  [ linear、fully-connected layer（with only one neuron）] - output（softmax） ， regard softmax as part of
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
from torch import optim  #在重构方法2、3的基础上改进训练过程
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
)   # 从numpy到tensor
n, c = x_train.shape

####################################
# 初始化：超参（weights的初始化在模型构造里定义了）
####################################
bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for，一个epoch有 训练样本数/bs 次循环，也就是，总共更新weights参数 epochs x 训练样本数 / bs 次


####################################
# 模型构造：用重构方法2
####################################
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))    # Xavier初始化，randn为正态随机，rand为普通的随机
        self.bias = nn.Parameter(torch.zeros(10))
    def forward(self, xb):
        return xb @ self.weights + self.bias
model = Mnist_Logistic()
loss_func = F.cross_entropy  # 因为是单层的网络，所以直接把激活+loss结合在了一起，就是交叉熵


####################################
# 训练过程
####################################
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()   # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()    # used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )   # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式(将元组解压为列表，才能应用numpy中的计算)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

# 封装之后，整个训练过程变为：
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

