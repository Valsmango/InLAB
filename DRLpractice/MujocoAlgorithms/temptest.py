import gym
import argparse
import os
import numpy as np
import torch
from matplotlib import pyplot as plt


env = gym.make("Hopper-v2")

env.seed(0)
env.action_space.seed(0)
torch.manual_seed(0)
np.random.seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

print(f"state_dim:{state_dim}; action_dim:{action_dim}; max_action:{max_action}")
print(env.action_space.low)

# position = np.zeros(action_dim)  # 设置初始位置
# walk = []
# steps = 10  # 设置步数
# for i in range(steps):
#     step = np.random.randint(0, 2, size=action_dim) * 2 - 1  # 如果随机值等于0则step为-1，反之为1
#     position += step  # 改变位置（正，负）
#     walk.append(position)

# walk = torch.tensor(walk)
# print(walk)
# print("--------------")
# rw = np.random.normal(0, max_action * 0.1, size=(10000, action_dim))
# # print(rw)
# print("-----rw---------")
# tt = np.cumsum(rw, axis = 0)*0.1
#
#
#
# # print("-----tt---------")
# # tt = np.random.normal(0, max_action * 0.1, size=(100, action_dim))
# plt.title("Random Walk-0")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.plot(range(10000), rw[:, 0])
# plt.plot(range(10000), tt[:, 0])
# plt.show()

# plt.title("Random Walk-1")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.plot(range(100), rw[:, 1])
# plt.plot(range(100), tt[:, 1])
# plt.show()
#
# plt.title("Random Walk-2")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.plot(range(100), rw[:, 2])
# plt.plot(range(100), tt[:, 2])
# plt.show()

for i in range(10):
    eps = np.random.random(1)

    print(i)
    print(eps)
    if i > 0 and eps < 0.3:
        print("yes")
    elif i > 0 and eps >= 0.3:
        print("no")