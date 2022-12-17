import collections
import itertools
import numpy as np
import torch
import time
from collections import namedtuple, deque


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

    t0 = time.time()
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(a.size()[0])
    time.sleep(0.1)
    t1 = time.time()
    print(f"运行时间为：{t1-t0:.2f}s")
