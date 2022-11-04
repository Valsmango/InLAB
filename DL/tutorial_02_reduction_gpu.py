"""this file is the note of [ What is torch.nn really? ], it introduces a 3-layer CNN model to classify photos in MNIST
dataset;    it is based on the analysis of [ tutorial_02.py ], and it constructs the network with nn.Sequential;
    Problem：multi-classification；
    Network Structure： [ conv（ activation： relu ） - conv（relu） - conv（relu） - avg_pooling ] - output（softmax）， it
        regards softmax as part of multi-classification process，which can be expressed by cross-entropy loss, rather
        than activation function；
    Loss func：cross_entropy；
    Device: GPU     ( focus on line 23-29、line 68、 line 140 )；
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

# device对象
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("cuda")
else:
    dev = torch.device("cpu")
    print("cpu")


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
# 初始化：超参
####################################
bs = 64  # batch size
lr = 0.1  # learning rate
epochs = 2  # how many epochs to train for


####################################
# 模型构造、函数封装：    Lambda类 作用于模型构造 nn.Sequential； preprocess 和 WrappedDataLoader类 作用于数据抽象:
####################################
# 改写2：在1的基础上修改图片的尺度：拓展适应性
class Lambda(nn.Module):    # 作用于模型构造，因为nn.Sequential只接受 a Module subclass
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x, y):   # 直接基于 dataloader 对象是features和labels对齐了的，所以不能只有x
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


class WrappedDataLoader:  # dl = dataloader 直接传数据； func 传入 preprocess 用来转换数据格式，并且可以方便调用GPU
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def get_model():
    model = nn.Sequential(    # 1中第一层数据转换变为了WrappedDataLoader中的一部分
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),     #虽然拓展了尺度，但是channels还是没有变
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        # replace nn.AvgPool2d with nn.AdaptiveAvgPool2d, which allows us to define the size of the output tensor we
        # want, rather than the input tensor we have.
        nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )     # 为什么用lambda x: x.view()？ 因为nn.Sequential自动传输每一层的输入输出，没有显性的可以直接调用的x，只有用lambda
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, opt



def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb).to(dev)
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
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
model, opt = get_model()
model.to(dev)
loss_func = F.cross_entropy
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


