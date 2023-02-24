# coding=utf-8
import numpy as np
import torch


class TD3ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.not_done = []

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
        state_sample = [self.state[i] for i in ind]
        action_sample = [self.action[i] for i in ind]
        next_state_sample = [self.next_state[i] for i in ind]
        reward_sample = [self.reward[i] for i in ind]
        not_done_sample = [self.not_done[i] for i in ind]
        return state_sample, action_sample, next_state_sample, reward_sample, not_done_sample

    def __len__(self) -> int:
        return len(self.done)


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