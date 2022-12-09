import collections
import itertools
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
    buffer = SubReplayBuffer(10)
    buffer.push('aaa', 18)
    buffer.push('bbb', 19)
    buffer.push('ccc', 20)
    # ans = buffer.sample(True, 2, 0)
    # print(ans.__getitem__(1))

    for i in range(len(buffer)):
        print(buffer.__getattribute__("age")[i])
