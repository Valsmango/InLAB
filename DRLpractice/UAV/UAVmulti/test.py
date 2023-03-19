# coding=utf-8
import numpy as np
import torch

class MADDPGReplayBuffer(object):
    def __init__(self, n_agents, state_dim, action_dim, max_size):
        self.n_agents = n_agents
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = [[]]
        self.action = [[]]
        self.next_state = [[]]
        self.reward = [[]]
        self.not_done = [[]]

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


if __name__ == "__main__":
    # observations = torch.tensor([np.array([1, 2]), np.array([2, 4]), np.array([3, 6])])
    # obs = torch.tensor([np.array([[1, 1, 1], [2, 2, 2]]),
    #                     np.array([[2, 1, 0], [5, 6, 14]]),
    #                     np.array([[3, 2, 1], [7, 0, 8]])])
    # tar = []
    # for i in range(3):
    #     tar.append(np.zeros([2, 2]))    # 最多5步
    # for i in range(3):
    #     tar[i][0] = np.stack(10.0 - observations[i, :])
    # print(tar)
    #
    # index = 0
    # print(index)
    # tt = [tar[agent_i][index] for agent_i in range(3)]
    # tt = torch.tensor(tt, dtype=torch.float)
    # print(tt)
    # r = 2500
    # alpha2 = np.random.rand() * 2 * np.pi
    # x2 = np.cos(alpha2) * r + 2500
    # y2 = np.sin(alpha2) * r + 2500
    # z2 = np.random.rand() * 300
    # print(f"x:{x2}  y:{y2}  z:{z2}")
    # dis = np.sqrt((x2-2500) ** 2 + (y2-2500) ** 2)
    # print(dis)
    # state = [torch.Tensor([[1,2,3,4], [5,6,7,8]]),torch.Tensor([[10,20,30,40], [50,60,70,80]])]
    # s = torch.cat(state, dim=1)
    s=float(True)
    print(s)