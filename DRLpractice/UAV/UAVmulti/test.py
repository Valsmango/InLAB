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
    # a = torch.tensor([[[-1],[2],[-3]],[[1],[-2],[3]]])
    # b = torch.tensor([[[-100], [-9], [10]], [[-8], [9], [-10]]])
    # print(torch.min(a,b))
    # done_bools = [float(True)]
    # print(done_bools)
    # for i in range(10):
    #     print(i%6)

    a = torch.tensor([[[0.0617, -0.0385, -0.1021],
                       [0.0802, -0.0787, -0.1001]],

                      [[0.1416, 0.0683, -0.0307],
                       [0.1131, 0.0756, 0.0281]],

                      [[-0.0831, -0.0678, 0.1418],
                       [-0.0507, -0.0399, 0.0688]]])
    s = torch.tensor([[[0.3686, 0.7281, 0.3159, -0.3612, 0.4827, 0.2590, 0.3214,
                        0.9426, 0.5364, 0.6628, 0.6151, 0.5941, 0.6677, 0.3320,
                        0.8236],
                       [0.2694, 0.1154, 0.1952, 0.1623, 0.4270, 0.4571, 0.7454,
                        0.9201, 0.8309, 0.8263, 0.8008, 0.3458, 0.9489, 0.4796,
                        0.5041]],

                      [[0.6628, 0.6151, 0.5941, 0.5349, 0.1964, -0.1160, 0.9034,
                        0.6876, 0.1735, 0.6677, 0.3320, 0.8236, 0.3686, 0.7281,
                        0.3159],
                       [0.8263, 0.8008, 0.3458, -0.4012, -0.2786, 0.2161, 0.1433,
                        0.1751, 0.6662, 0.9489, 0.4796, 0.5041, 0.2694, 0.1154,
                        0.1952]],

                      [[0.6677, 0.3320, 0.8236, 0.3966, -0.3051, 0.6885, 0.9494,
                        0.5487, 0.7062, 0.3686, 0.7281, 0.3159, 0.6628, 0.6151,
                        0.5941],
                       [0.9489, 0.4796, 0.5041, -0.4190, 0.0422, 0.1326, 0.0164,
                        0.5248, 0.5035, 0.2694, 0.1154, 0.1952, 0.8263, 0.8008,
                        0.3458]]])
    # s_a = torch.cat([s, a], dim=2)
    # s_a = s_a.permute(1, 0, 2)
    # # s_a = s_a.view(-1, 3*18)
    # s_a = s_a.reshape(-1, 3*18)
    # print(s_a)
