# coding=utf-8
import numpy as np
import torch


class PPODiscreteReplayBuffer:
    def __init__(self, batch_size):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []
        self.batch_size = batch_size

    def put(self, s, a, r, done, p, v):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(done)
        self.prob.append(p)
        self.value.append(v)

    def sample(self):
        batch_step = np.arange(0, len(self.done), self.batch_size)  # np.arange(start = 0, stop = len, step = batch_sz) 即： 【0， batch_sz, batch_sz*2, ... < stop】
        indicies = np.arange(len(self.done), dtype=np.int64)  # np.arange(start = 0, stop = len, step = 1)
        np.random.shuffle(indicies)     # 将indicies弄成乱序
        batches = [indicies[i:i+self.batch_size] for i in batch_step]
        return self.state, self.action, self.reward, self.done, self.prob, self.value, batches

    def clear(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []

    def __len__(self) -> int:
        return len(self.done)


class TD3ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # self.state = np.zeros((max_size, state_dim))
        # self.action = np.zeros((max_size, action_dim))
        # self.next_state = np.zeros((max_size, state_dim))
        # self.reward = np.zeros((max_size, 1))
        # self.not_done = np.zeros((max_size, 1))
        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.not_done = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        if self.ptr < self.max_size:
            self.state.append(state[0])
            self.action.append(action[0])
            self.next_state.append(next_state[0])
            self.reward.append(reward)
            self.not_done.append(1. - done)
        else:
            self.state[self.ptr] = state[0]
            self.action[self.ptr] = action[0]
            self.next_state[self.ptr] = next_state[0]
            self.reward[self.ptr] = reward
            self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        state_sample= [self.state[i] for i in ind]
        action_sample = [self.action[i] for i in ind]
        next_state_sample = [self.next_state[i] for i in ind]
        reward_sample = [self.reward[i] for i in ind]
        not_done_sample = [self.not_done[i] for i in ind]
        # return (
        #     torch.FloatTensor(self.state[ind]).to(self.device),
        #     torch.FloatTensor(self.action[ind]).to(self.device),
        #     torch.FloatTensor(self.next_state[ind]).to(self.device),
        #     torch.FloatTensor(self.reward[ind]).to(self.device),
        #     torch.FloatTensor(self.not_done[ind]).to(self.device)
        # )
        return state_sample, action_sample, next_state_sample, reward_sample, not_done_sample