import collections
import itertools
import numpy as np
import torch
import time
import gym
from collections import namedtuple, deque
import DRLpractice.GymAlgorithms.plotUtils.PlotUtils as pltutil


class SubReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, capacity):
        self.Transition = namedtuple("User",
                        ("name", "age"))
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, random_update=False, lookup_step=None, idx=None):
        if random_update is False:
            return self.memory
        else:
            # return collections.deque(itertools.islice(self.memory, idx, idx + lookup_step))
            return self.memory.__getitem__()    # 用slice!!!!!取[idx:idx + lookup_step]切片

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    # q_taget = torch.zeros([2, 3, 4])
    # q_taget[0][0][0] = 0
    # q_taget[0][1][0] = 1
    # q_taget[0][2][0] = 2
    # q_taget[1][0][0] = 5
    # q_taget[1][1][0] = 6
    # q_taget[1][2][0] = 7
    # q_taget[0][0][1] = 1
    # q_taget[0][1][1] = 2
    # q_taget[0][2][1] = 3
    # q_taget[1][0][1] = 4
    # q_taget[1][1][1] = 5
    # q_taget[1][2][1] = 6
    # print(q_taget.max(2)[1])

    # t0 = time.time()
    # a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(a.size()[0])
    # time.sleep(0.1)
    # t1 = time.time()
    # print(f"运行时间为：{t1-t0:.2f}s")

    # batch_step = np.arange(0, 10,
    #                        4)  # np.arange(start = 0, stop = len, step = batch_sz) 即： 【0， batch_sz, batch_sz*2, ... < stop】
    # indicies = np.arange(10, dtype=np.int64)  # np.arange(start = 0, stop = len, step = 1)
    # print(indicies)     # [0 1 2 3 4 5 6 7 8 9]
    # np.random.shuffle(indicies)  # 将indicies弄成乱序
    # print(indicies)     # [8 7 9 6 0 2 3 4 5 1]
    # batches = [indicies[i:i + 4] for i in batch_step]
    # print(batches)       # [array([8, 7, 9, 6], dtype=int64), array([0, 2, 3, 4], dtype=int64), array([5, 1], dtype=int64)]

    # rewards_base = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.2, 3.3, 4.4, 5.5], [0.9, 1.9, 3.5, 4.5, 12]]
    # pltutil.plot_one_alg_rewards(rewards=rewards_base, algo_name="Algo-1")

    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n
    try:
        n_states = env.observation_space.n
    except AttributeError:
        n_states = env.observation_space.shape[0]
    print(f"action space:{n_actions}, state space:{n_states}")