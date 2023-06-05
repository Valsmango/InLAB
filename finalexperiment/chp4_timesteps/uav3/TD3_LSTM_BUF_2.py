# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy
import os
import time
from typing import Dict, List, Tuple
import collections
import sys
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Env(object):
    def __init__(self):
        self.success_count = 0
        # self.collision_count = 0
        self.uav_state = []
        self.path = []
        self.delta_t = 1
        self.min_sep_hori = 152.4
        self.min_sep_vert = 30.48
        self.min_range = 1000
        self.viewer = None
        self._max_episode_steps = 200
        self.marker_list = ['x', '.', '|']
        # self.color_list = ['green', 'blue', 'red']

        self.max_action = np.array([10.0, 10.0, 2.0])
        # ndarray - tuple
        self.n_uav = 3
        self.continue_flag = [True for _ in range(self.n_uav)]

        # assert (mode == "train" or mode == "test"), "Env Mode Error！"
        # # 随机起点和终点
        # if mode == "train" or mode == "test":
        r1 = np.random.rand() * 300 + 2200
        alpha1 = np.random.rand() * 2 * np.pi
        x1 = np.cos(alpha1) * r1 + 2500
        y1 = np.sin(alpha1) * r1 + 2500
        z1 = np.random.rand() * 200 + 50
        tar_x1 = 5000 - x1
        tar_y1 = 5000 - y1
        tar_z1 = 300 - z1
        v_x1 = (tar_x1 - x1) / 50
        v_y1 = (tar_y1 - y1) / 50
        v_z1 = (tar_z1 - z1) / 50
        r2 = np.random.rand() * 300 + 2200
        alpha2 = np.random.rand() * 2 * np.pi
        x2 = np.cos(alpha2) * r2 + 2500
        y2 = np.sin(alpha2) * r2 + 2500
        z2 = np.random.rand() * 200 + 50
        tar_x2 = 5000 - x2
        tar_y2 = 5000 - y2
        tar_z2 = 300 - z2
        v_x2 = (tar_x2 - x2) / 50
        v_y2 = (tar_y2 - y2) / 50
        v_z2 = (tar_z2 - z2) / 50
        r3 = np.random.rand() * 300 + 2200
        alpha3 = np.random.rand() * 2 * np.pi
        x3 = np.cos(alpha3) * r3 + 2500
        y3 = np.sin(alpha3) * r3 + 2500
        z3 = np.random.rand() * 200 + 50
        tar_x3 = 5000 - x3
        tar_y3 = 5000 - y3
        tar_z3 = 300 - z3
        v_x3 = (tar_x3 - x3) / 50
        v_y3 = (tar_y3 - y3) / 50
        v_z3 = (tar_z3 - z3) / 50
        init_start = [
            [x1, y1, z1, v_x1, v_y1, v_z1, tar_x1, tar_y1, tar_z1,
             x2, y2, z2, x3, y3, z3],
            [x2, y2, z2, v_x2, v_y2, v_z2, tar_x2, tar_y2, tar_z2,
             x1, y1, z1, x3, y3, z3],
            [x3, y3, z3, v_x3, v_y3, v_z3, tar_x3, tar_y3, tar_z3,
             x1, y1, z1, x2, y2, z2]]
        init_target = [[tar_x1, tar_y1, tar_z1], [tar_x2, tar_y2, tar_z2], [tar_x3, tar_y3, tar_z3]]

            # init_start = [
            #     [200, 2500, 150, (4800-200)/50, 0, 0, 4800, 2500, 150,
            #      2500, 200, 150, 2500+2300/np.sqrt(2), 2500+2300/np.sqrt(2), 150],
            #     [2500, 200, 150, 0, (4800-200)/50, 0, 2500, 4800, 150,
            #      200, 2500, 150, 2500+2300/np.sqrt(2), 2500+2300/np.sqrt(2), 150],
            #     [2500+2300/np.sqrt(2), 2500+2300/np.sqrt(2), 150, (-4600/np.sqrt(2))/50, (-4600/np.sqrt(2))/50, 0,
            #      2500-2300/np.sqrt(2), 2500-2300/np.sqrt(2), 150,
            #      2500, 200, 150, 200, 2500, 150]]
            # init_target = [[4800, 2500, 150], [2500, 4800, 150],[2500-2300/np.sqrt(2), 2500-2300/np.sqrt(2), 150]]

        self._init_map(init_start, init_target)

    def _init_map(self, init_start, init_target):
        self.target = init_target
        self.uav_init_state = init_start
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.att = 0.0025 * 2 / ((self.target[0][0] / 1000 - self.uav_init_state[0][0] / 1000) ** 2 +
                                 (self.target[0][1] / 1000 - self.uav_init_state[0][1] / 1000) ** 2 +
                                 (self.target[0][2] / 1000 - self.uav_init_state[0][2] / 1000) ** 2)
        self.rep = 0.00125 * 2 / (1 / self.min_sep_hori - 1 / self.min_range) ** 2
        # print(f"att parameter: {self.att}; rep parameter: {self.rep}")
        for i in range(self.n_uav):
            self.path.append([[self.uav_init_state[i][0], self.uav_init_state[i][1], self.uav_init_state[i][2]]])

    def reset(self):
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.continue_flag = [True for _ in range(self.n_uav)]
        for i in range(self.n_uav):
            self.path.append([[self.uav_init_state[i][0], self.uav_init_state[i][1], self.uav_init_state[i][2]]])
        # 【x，y，z，v_x，v_y，v_z，tar_x，tar_y，tar_z，agent2_x，agent2_y，agent2_z】
        return_state = []
        for i in range(self.n_uav):
            return_state.append(np.array(self.uav_state[i]) / np.array([5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0, 5000.0, 5000.0, 300.0,
                                                                5000.0, 5000.0, 300.0, 5000.0, 5000.0, 300.0]))
        return torch.Tensor(return_state)

    def step(self, input_action):
        action = [np.array(ac) * self.max_action for ac in input_action]
        # action noise
        action += np.random.normal(loc=0, scale=0.003, size=len(self.max_action))
        done = [False for _ in range(self.n_uav)]
        reward = [0.0 for _ in range(self.n_uav)]
        store_flag = [False for _ in range(self.n_uav)]
        # pre_tar_dis = [0.0 for _ in range(self.n_uav)]
        # pre_tar_dis_hori = [0.0 for _ in range(self.n_uav)]
        # pre_tar_dis_vert = [0.0 for _ in range(self.n_uav)]
        tar_dis = [0.0 for _ in range(self.n_uav)]
        tar_dis_hori = [0.0 for _ in range(self.n_uav)]
        tar_dis_vert = [0.0 for _ in range(self.n_uav)]
        return_state = []

        # for i in range(self.n_uav):
        #     if self.continue_flag[i]:
        #         pre_tar_dis_hori[i] = np.sqrt((self.uav_state[i][0] - self.target[i][0]) ** 2 +
        #                                       (self.uav_state[i][1] - self.target[i][1]) ** 2)
        #         pre_tar_dis_vert[i] = np.abs(self.uav_state[i][2] - self.target[i][2])
        #         pre_tar_dis[i] = np.sqrt((self.uav_state[i][0] - self.target[i][0]) ** 2 +
        #                                (self.uav_state[i][1] - self.target[i][1]) ** 2 +
        #                                (self.uav_state[i][2] - self.target[i][2]) ** 2)

        # Calculate the uav's position
        for i in range(self.n_uav):
            if self.continue_flag[i]:
                store_flag[i] = True
                self.uav_state[i][3] += action[i][0]
                if self.uav_state[i][3] > 200:
                    self.uav_state[i][3] = 200
                    reward[i] += -1
                elif self.uav_state[i][3] < -200:
                    self.uav_state[i][3] = -200
                    reward[i] += -1
                self.uav_state[i][0] += self.uav_state[i][3] * self.delta_t
                self.uav_state[i][4] += action[i][1]
                if self.uav_state[i][4] > 200:
                    self.uav_state[i][4] = 200
                    reward[i] += -1
                elif self.uav_state[i][4] < -200:
                    self.uav_state[i][4] = -200
                    reward[i] += -1
                self.uav_state[i][1] += self.uav_state[i][4] * self.delta_t
                self.uav_state[i][5] += action[i][2]
                if self.uav_state[i][5] > 10:
                    self.uav_state[i][5] = 10
                    reward[i] += -1
                elif self.uav_state[i][5] < -10:
                    self.uav_state[i][5] = -10
                    reward[i] += -1
                self.uav_state[i][2] += self.uav_state[i][5] * self.delta_t
                self.path[i].append([self.uav_state[i][0], self.uav_state[i][1], self.uav_state[i][2]])
            else:
                self.uav_state[i] = [0. for _ in range(15)]

        # 判断是否存在冲突
        for i in range(self.n_uav):
            if store_flag[i]:
                j = i + 1
                while j < self.n_uav:
                    if store_flag[j]:
                        uav_hori_dis = np.sqrt((self.uav_state[i][0] - self.uav_state[j][0]) ** 2 +
                                               (self.uav_state[i][1] - self.uav_state[j][1]) ** 2)
                        uav_vert_dis = np.abs(self.uav_state[i][2] - self.uav_state[j][2])
                        uav_dis = np.sqrt((self.uav_state[i][0] - self.uav_state[j][0]) ** 2 +
                                          (self.uav_state[i][1] - self.uav_state[j][1]) ** 2 +
                                          (self.uav_state[i][2] - self.uav_state[j][2]) ** 2)

                        if uav_hori_dis < self.min_sep_hori and uav_vert_dis < self.min_sep_vert:
                            reward[i] += -100
                            reward[j] += -100
                            # self.collision_count += 1
                            done[i] = True
                            done[j] = True
                            self.continue_flag[i] = False
                            self.continue_flag[j] = False
                        elif uav_dis < self.min_range:
                            ################### 斥力计算方式1 ###################
                            urep = - 1 / 2 * self.rep * (1 / uav_dis - 1 / self.min_range)
                            reward[i] += urep
                            reward[j] += urep

                    j = j + 1

        for i in range(self.n_uav):
            if store_flag[i]:
                tar_dis[i] = np.sqrt((self.uav_state[i][0] - self.target[i][0]) ** 2 +
                                       (self.uav_state[i][1] - self.target[i][1]) ** 2 +
                                       (self.uav_state[i][2] - self.target[i][2]) ** 2)
                tar_dis_hori[i] = np.sqrt((self.uav_state[i][0] - self.target[i][0]) ** 2 +
                                            (self.uav_state[i][1] - self.target[i][1]) ** 2)
                tar_dis_vert[i] = np.abs(self.uav_state[i][2] - self.target[i][2])

                ################### 引力计算方式1 ###################
                uatt = - self.att * np.array(tar_dis[i])
                reward[i] += uatt

                # #################### 引力计算方式2 ###################
                # uatt = np.exp(- tar_dis_hori[i]/1000) + np.exp(- tar_dis_vert[i]/60)
                # reward[i] += uatt

                # ################### 引力计算方式3 ###################
                # # reward[i] += (pre_tar_dis[i] - tar_dis[i])/100  # 处理目标不可达
                # reward[i] += (pre_tar_dis_hori[i] - tar_dis_hori[i]) / 100 + \
                #              (pre_tar_dis_vert[i] - tar_dis_vert[i]) / 30

                reward[i] += -0.5  # 这一信息让其快速到达终点

                if not done[i]:
                    if tar_dis_vert[i] < self.min_sep_vert and tar_dis_hori[i] < self.min_sep_hori:
                        reward[i] += 100
                        self.success_count += 1
                        # reward[i] += 1
                        done[i] = True
                        self.continue_flag[i] = False
                    elif np.sqrt((self.uav_state[i][0] - 2500) ** 2 + (self.uav_state[i][1] - 2500) ** 2) > 2500 or \
                            self.uav_state[i][2] < 0.0 or self.uav_state[i][2] > 300.0:
                        reward[i] += -100
                        # reward[i] += -1
                        done[i] = True
                        self.continue_flag[i] = False
                    elif len(self.path[i]) > self._max_episode_steps:
                        done[i] = True
                        self.continue_flag[i] = False
            else:
                done[i] = True

        # 更新位置信息
        for i in range(self.n_uav):
            for j in range(self.n_uav-1):
                for k in range(3):
                    self.uav_state[i][9 + j * 3 + k] = self.uav_state[(i + j + 1) % self.n_uav][k]  # 0-8

            return_state.append(np.array(self.uav_state[i]) / np.array([5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0, 5000.0, 5000.0, 300.0,
                                                                5000.0, 5000.0, 300.0, 5000.0, 5000.0, 300.0]))

        return torch.Tensor(return_state), torch.Tensor(reward), done, store_flag

    # def render(self):
    #     if self.viewer is None:
    #         self.viewer = Viewer(self.uav_state, self.target)
    #     self.viewer.render()

    def sample_action(self):
        # # Mean
        return_action = []
        for i in range(self.n_uav):
            random_delta_v_x = np.random.rand() * 2 - 1
            random_delta_v_y = np.random.rand() * 2 - 1
            random_delta_v_z = np.random.rand() * 2 - 1
            return_action.append([random_delta_v_x, random_delta_v_y, random_delta_v_z])
        return torch.Tensor(return_action)

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def show_path(self):
        pass

    def close(self):
        if self.viewer:
            # self.viewer.close()
            self.viewer = None


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.hidden_width = hidden_width
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.LSTM(hidden_width, hidden_width, batch_first=True)
        self.l4 = nn.Linear(hidden_width, action_dim)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3, gain=0.01)

    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_width]), torch.zeros([1, batch_size, self.hidden_width])
        else:
            return torch.zeros([1, 1, self.hidden_width]), torch.zeros([1, 1, self.hidden_width])

    def forward(self, s, h_0, c_0):
        # https://blog.csdn.net/feifei3211/article/details/102998288?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-102998288-blog-109586782.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-102998288-blog-109586782.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=1
        if not hasattr(self, '_flattened'):
            self.l3.flatten_parameters()
            setattr(self, '_flattened', True)
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        s, (h_t, c_t) = self.l3(s, (h_0, c_0))
        a = self.max_action * torch.tanh(self.l4(s))  # [-max,max]
        return a, h_t, c_t


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.hidden_width = hidden_width
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.LSTM(hidden_width, hidden_width, batch_first=True)
        self.l4 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3)
        # Q2
        self.l5 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l6 = nn.Linear(hidden_width, hidden_width)
        self.l7 = nn.LSTM(hidden_width, hidden_width, batch_first=True)
        self.l8 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l4)
        # orthogonal_init(self.l5)
        # orthogonal_init(self.l6)

    def init_hidden_state(self, batch_size, training=None):
        '''
        https://www.cnblogs.com/jiangkejie/p/13246857.html
        '''
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_width]), torch.zeros([1, batch_size, self.hidden_width])
        else:
            return torch.zeros([1, 1, self.hidden_width]), torch.zeros([1, 1, self.hidden_width])

    def forward(self, s, a, h_0_1, c_0_1, h_0_2, c_0_2):
        if not hasattr(self, '_flattened'):
            self.l3.flatten_parameters()
            self.l7.flatten_parameters()
            setattr(self, '_flattened', True)
        # s_a = torch.cat([s, a], 1)
        s_a = torch.cat([s, a], 2)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1, (h_t_1, c_t_1) = self.l3(q1, (h_0_1, c_0_1))  # q1 = F.relu(self.l1(s_a))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(s_a))
        q2 = F.relu(self.l6(q2))
        q2, (h_t_2, c_t_2) = self.l7(q2, (h_0_2, c_0_2))   # q2 = F.relu(self.l4(s_a))
        q2 = self.l8(q2)

        return q1, q2, h_t_1, c_t_1, h_t_2, c_t_2

    def Q1(self, s, a, h_0_1, c_0_1):
        if not hasattr(self, '_flattened'):
            self.l3.flatten_parameters()
            setattr(self, '_flattened', True)
        # s_a = torch.cat([s, a], 1)
        s_a = torch.cat([s, a], 2)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1, (h_t_1, c_t_1) = self.l3(q1, (h_0_1, c_0_1))  # q1 = F.relu(self.l1(s_a))
        q1 = self.l4(q1)

        return q1, h_t_1, c_t_1


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                 max_epi_num=20000, max_epi_len=200,
                 batch_size=1,
                 lookup_step=None):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit(
                'It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        # self.memory = collections.deque(maxlen=self.max_epi_num)
        self.failure_size = int(self.max_epi_num * 0.5)
        self.success_size = int(self.max_epi_num - self.failure_size)
        self.avg_reward = 0.
        self.success_memory = collections.deque(maxlen=self.success_size)
        self.success_count = 0
        self.failure_memory = collections.deque(maxlen=self.failure_size)
        self.failure_count = 0
        self.total_count = 0

    def put(self, episode, episode_reward):
        # self.memory.append(episode)
        if episode_reward > self.avg_reward:
            self.success_memory.append(episode)
            self.success_count = min(self.success_count + 1, self.success_size)
        else:
            self.failure_memory.append(episode)
            self.failure_count = min(self.failure_count + 1, self.failure_size)
        self.total_count += 1
        self.avg_reward += (episode_reward - self.avg_reward) / self.total_count

    def _pre_sample(self, ratio):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random upodate
            sampled_episodes = []
            success_num = int(self.batch_size * ratio[0])
            failure_num = self.batch_size
            if success_num > 0:
                success_count = min(success_num, self.success_count)
                sampled_episodes.extend(random.sample(self.success_memory, success_count))
                failure_num -= success_num
            sampled_episodes.extend(random.sample(self.failure_memory, failure_num))


            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def sample(self, batch_size, ratio=[0.0, 0.0]):
        samples, seq_len = self._pre_sample(ratio)
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for i in range(batch_size):
            observations.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            rewards.append(samples[i]["rews"])
            next_observations.append(samples[i]["next_obs"])
            dones.append(samples[i]["done"])

        observations = torch.FloatTensor(observations).reshape(batch_size, seq_len, -1).to(device)
        actions = torch.FloatTensor(actions).reshape(batch_size, seq_len, -1).to(device)
        rewards = torch.FloatTensor(rewards).reshape(batch_size, seq_len, -1).to(device)
        next_observations = torch.FloatTensor(next_observations).reshape(batch_size, seq_len, -1).to(device)
        dones = torch.FloatTensor(dones).reshape(batch_size, seq_len, -1).to(device)
        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        if transition[5]:
            self.obs.append(np.array(transition[0]))
            self.action.append(np.array(transition[1]))
            self.reward.append(np.array(transition[2]))
            self.next_obs.append(np.array(transition[3]))
            self.done.append(np.array(transition[4]))

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx + lookup_step]
            action = action[idx:idx + lookup_step]
            reward = reward[idx:idx + lookup_step]
            next_obs = next_obs[idx:idx + lookup_step]
            done = done[idx:idx + lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)



class TD3LSTM(object):
    def __init__(self, batch_size, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = batch_size  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.policy_noise = 0.2 * max_action  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5 * max_action  # Clip the noise
        self.policy_freq = 2  # The frequency of policy updates
        self.actor_pointer = 0

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, h, c):
        s = torch.FloatTensor(s.reshape(1, -1)).unsqueeze(0).to(device) # 输入的s【12】，然后变成了【1，1，12】，而h【1，1，256】
        h = torch.FloatTensor(h).to(device)
        c = torch.FloatTensor(c).to(device)
        a, h, c = self.actor(s, h,  c)
        return a.cpu().data.numpy().flatten(), h.cpu().data.numpy(), c.cpu().data.numpy()

    def learn(self, relay_buffer):
        self.actor_pointer += 1

        buffer_ratio = [0.6, 0.0]
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size, buffer_ratio)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            h_target_0, c_target_0 = self.actor.init_hidden_state(batch_size=self.batch_size, training=True)
            h_target_0, c_target_0 = h_target_0.to(device), c_target_0.to(device)
            h_target_1, c_target_1 = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
            h_target_2, c_target_2 = copy.deepcopy(h_target_1), copy.deepcopy(c_target_1)
            h_target_1, c_target_1 = h_target_1.to(device), c_target_1.to(device)
            h_target_2, c_target_2 = h_target_2.to(device), c_target_2.to(device)
            # Trick 1:target policy smoothing
            # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, h_target_0, c_target_0 = self.actor_target(batch_s_, h_target_0, c_target_0)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            # Trick 2:clipped double Q-learning
            target_Q1, target_Q2, h_target_1, c_target_1, h_target_2, c_target_2 = \
                self.critic_target(batch_s_, next_action, h_target_1, c_target_1, h_target_2, c_target_2)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)

        # Get the current Q
        h_current, c_current = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current, c_current = h_current.to(device), c_current.to(device)
        h_current_, c_current_ = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current_, c_current_ = h_current_.to(device), c_current_.to(device)
        current_Q1, current_Q2, h_current, c_current, h_current_, c_current_ = \
            self.critic(batch_s, batch_a, h_current,c_current, h_current_, c_current_)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_freq == 0:
            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            # Compute actor loss
            h, c = self.actor.init_hidden_state(batch_size=batch_size, training=True)
            h, c = h.to(device), c.to(device)
            a, h, c = self.actor(batch_s, h, c)
            h_1, c_1 = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
            h_1, c_1 = h_1.to(device), c_1.to(device)
            actor_loss, h_1, c_1 = self.critic.Q1(batch_s, a, h_1, c_1)  # Only use Q1
            actor_loss = -actor_loss.mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic networks
            for params in self.critic.parameters():
                params.requires_grad = True

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

if __name__ == '__main__':
    start_time = time.time()
    env_name ="MAEnv"
    model_name = "TD3_LSTM_BUF"
    # Set random seed
    seed = 10
    # env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    state_dim = 15
    action_dim = 3
    max_action = 1.0
    n_agents = 3
    max_episode_steps = 200  # Maximum number of steps per episode

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 3e6  # Maximum number of training steps
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    max_train_episodes = 2e4

    batch_size = 64  # random --> 64, sequential --> 1
    lookup_step = 16

    agent = [TD3LSTM(batch_size, state_dim, action_dim, max_action) for _ in range(n_agents)]
    replay_buffer = [EpisodeMemory(random_update=True,
                                  max_epi_num=20000, max_epi_len=200,
                                   batch_size=batch_size,
                                   lookup_step=lookup_step) for _ in range(n_agents)]

    if not os.path.exists(f"./reward_train/{model_name}/{env_name}"):
        os.makedirs(f"./reward_train/{model_name}/{env_name}")
    if not os.path.exists(f"./model_train/{model_name}/{env_name}"):
        os.makedirs(f"./model_train/{model_name}/{env_name}")

    env = Env()
    s, done = env.reset(), [False for _ in range(n_agents)]

    episode_record = [EpisodeBuffer() for _ in range(n_agents)]
    h = []
    c = []

    for i in range(n_agents):
        h_i, c_i = agent[i].actor.init_hidden_state(batch_size=batch_size, training=False)
        h.append(h_i)
        c.append(c_i)

    episode_reward_agents = [0 for _ in range(n_agents)]
    train_episode_rewards = []
    episode_reward = 0.
    episode_timesteps = 0
    episode_num = 0
    train_episode_success_rate = []
    time_records = []

    for t in tqdm(range(int(max_train_steps))):

        episode_timesteps += 1

        if t < random_steps:  # Take random actions in the beginning for the better exploration
            a = env.sample_action()
        else:
            a = []
            for i in range(n_agents):
                a_i, h[i], c[i]= agent[i].choose_action(s[i], h[i], c[i])
                a_i = (a_i + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                a.append(a_i)

        s_, r, done, store_flags = env.step(a)
        for i in range(n_agents):
            dw = float(done[i]) if episode_timesteps < env._max_episode_steps else 0
            episode_record[i].put([s[i], a[i], r[i], s_[i], dw, store_flags[i]])  # Store the transition
        s = s_
        episode_reward += sum(r)
        for i in range(n_agents):
            episode_reward_agents[i] += r[i]

        if t >= random_steps:
            for i in range(n_agents):
                agent[i].learn(replay_buffer[i])

        if np.all(np.array(done)):
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            time_records.append(t)
            train_episode_rewards.append(episode_reward)
            train_episode_success_rate.append(env.success_count/n_agents)
            if (episode_num + 1) % 100 == 0:
                np.save(f'./reward_train/{model_name}/{env_name}/timestep_seed_{seed}.npy', np.array(time_records))
                np.save(f'./reward_train/{model_name}/{env_name}/reward_seed_{seed}.npy', np.array(train_episode_rewards))
                np.save(f'./reward_train/{model_name}/{env_name}/success_seed_{seed}.npy', np.array(train_episode_success_rate))
                for i in range(n_agents):
                    np.save(f'./reward_train/{model_name}/{env_name}/agent_{i}_reward_seed_{seed}.npy', np.array(episode_reward_agents[i]))

            # Reset environment
            for i in range(n_agents):
                replay_buffer[i].put(episode_record[i], episode_reward_agents[i])
            env.close()
            env = Env()
            s, done = env.reset(), [False for i in range(n_agents)]
            episode_reward = 0.
            episode_reward_agents = [0. for _ in range(n_agents)]
            episode_timesteps = 0
            episode_num += 1

            episode_record = [EpisodeBuffer() for _ in range(n_agents)]
            h = []
            c = []
            for i in range(n_agents):
                h_i, c_i = agent[i].actor.init_hidden_state(batch_size=batch_size, training=False)
                h.append(h_i)
                c.append(c_i)

            if (episode_num + 1) % 5000 == 0:
                for i in range(n_agents):
                    agent[i].save(f"./model_train/{model_name}/{env_name}/agent_{i}_{model_name}")


    env.close()
    for i in range(n_agents):
        agent[i].save(f"./model_train/{model_name}/{env_name}/agent_{i}_{model_name}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}")