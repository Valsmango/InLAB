# coding=utf-8
import torch
import numpy as np
import copy

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

        # # 随机起点和终点
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

