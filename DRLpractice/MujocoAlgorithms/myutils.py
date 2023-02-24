import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
        # 为了方便每一步的sample采用replay_buffer or high_replay_buffer而增加的,但是存储high_replay_buffer过于麻烦了
        # return self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind]


class PPOReplayBuffer(object):
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []

    def add(self, s, a, r, done, p, v):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(done)
        self.prob.append(p)
        self.value.append(v)

    def clear(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []

    def sample(self, batch_size):
        batch_step = np.arange(0, len(self.done), batch_size)  # np.arange(start = 0, stop = len, step = batch_sz) 即： 【0， batch_sz, batch_sz*2, ... < stop】
        indicies = np.arange(len(self.done), dtype=np.int64)  # np.arange(start = 0, stop = len, step = 1)
        np.random.shuffle(indicies)     # 将indicies弄成乱序
        batches = [indicies[i:i+batch_size] for i in batch_step]
        return self.state, self.action, self.reward, self.done, self.prob, self.value, batches

    def __len__(self) -> int:
        return len(self.done)