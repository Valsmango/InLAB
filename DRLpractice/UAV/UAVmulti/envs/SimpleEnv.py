# coding=utf-8
import time
import pyglet
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import copy
import torch

'''
simple env
'''


class Env(object):
    def __init__(self, mode="Train"):
        self.success_count = 0
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
        self.n_uav = 2
        self.continue_flag = [True for _ in range(self.n_uav)]

        assert (mode == "train" or mode == "test"), "Env Mode Error！"
        # 随机起点和终点
        if mode == "train" or mode == "test":
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
            init_start = [
                [x1, y1, z1, v_x1, v_y1, v_z1, tar_x1, tar_y1, tar_z1,
                 x2, y2, z2, v_x2, v_y2, v_z2],
                [x2, y2, z2, v_x2, v_y2, v_z2, tar_x2, tar_y2, tar_z2,
                 x1, y1, z1, v_x1, v_y1, v_z1]]
            init_target = [[tar_x1, tar_y1, tar_z1], [tar_x2, tar_y2, tar_z2]]

            # init_start = [
            #     [0, 2500, 150, 100, 0, 0, 5000, 2500, 150,
            #      2500, 0, 150, 0, 100, 0],
            #     [2500, 0, 150, 0, 100, 0, 2500, 5000, 150,
            #      0, 2500, 150, 100, 0, 0]]
            # init_target = [[5000, 2500, 150], [2500, 5000, 150]]

        self._init_map(init_start, init_target)

    def reset(self):
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.continue_flag = [True for _ in range(self.n_uav)]
        for i in range(self.n_uav):
            self.path.append([[self.uav_init_state[i][0], self.uav_init_state[i][1], self.uav_init_state[i][2]]])
        # 【x，y，z，v_x，v_y，v_z，tar_x，tar_y，tar_z，agent2_x，agent2_y，agent2_z】
        return_state = [np.array(self.uav_state[0]) / np.array([5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0, 5000.0, 5000.0, 300.0,
                                                                5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0]),
                        np.array(self.uav_state[1]) / np.array([5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0, 5000.0, 5000.0, 300.0,
                                                                5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0])]
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

        # 判断是否存在冲突
        for i in range(self.n_uav):
            if self.continue_flag[i]:
                j = i + 1
                while j < self.n_uav:
                    if self.continue_flag[j]:
                        uav_hori_dis = np.sqrt((self.uav_state[i][0] - self.uav_state[j][0]) ** 2 +
                                               (self.uav_state[i][1] - self.uav_state[j][1]) ** 2)
                        uav_vert_dis = np.abs(self.uav_state[i][2] - self.uav_state[j][2])
                        uav_dis = np.sqrt((self.uav_state[i][0] - self.uav_state[j][0]) ** 2 +
                                          (self.uav_state[i][1] - self.uav_state[j][1]) ** 2 +
                                          (self.uav_state[i][2] - self.uav_state[j][2]) ** 2)
                        if uav_dis < self.min_range:
                            ################### 斥力计算方式1 ###################
                            urep = - 1 / 2 * self.rep * (1 / uav_dis - 1 / self.min_range)
                            reward[i] += urep
                            reward[j] += urep

                            # #################### 斥力计算方式2 ###################
                            # urep = -np.exp(- uav_hori_dis / 100) -np.exp(- uav_vert_dis)
                            # reward[i] += urep
                            # reward[j] += urep

                        if uav_hori_dis < self.min_sep_hori and uav_vert_dis < self.min_sep_vert:
                            reward[i] += -100
                            reward[j] += -100
                            done[i] = True
                            done[j] = True
                            self.continue_flag[i] = False

                    j = j + 1

        for i in range(self.n_uav):
            if self.continue_flag[i]:
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
            for j in range(6):
                self.uav_state[i][j+9] = self.uav_state[(i + 1) % self.n_uav][j] # 0-8

            return_state.append(np.array(self.uav_state[i]) / np.array([5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0, 5000.0, 5000.0, 300.0,
                                                                5000.0, 5000.0, 300.0, 200.0, 200.0, 10.0]))

        return torch.Tensor(return_state), torch.Tensor(reward), done, store_flag

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.uav_state, self.target)
        self.viewer.render()

    def sample_action(self):
        # # Mean
        return_action = []
        for i in range(self.n_uav):
            random_delta_v_x = np.random.rand() * 2 - 1
            random_delta_v_y = np.random.rand() * 2 - 1
            random_delta_v_z = np.random.rand() * 2 - 1
            return_action.append([random_delta_v_x, random_delta_v_y, random_delta_v_z])
        return torch.Tensor(return_action)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def show_path(self):
        self._show_3D_path()
        self._show_xy_path()
        self._show_xz_path()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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

    def _show_3D_path(self):
        ax = plt.axes(projection='3d')

        for i in range(self.n_uav):
            path_len = len(self.path[i])
            x = [self.path[i][j][0] / 1000 for j in range(path_len)]
            y = [self.path[i][j][1] / 1000 for j in range(path_len)]
            z = [self.path[i][j][2] / 1000 for j in range(path_len)]
            ax.scatter([self.path[i][path_len - 1][0] / 1000], [self.path[i][path_len - 1][1] / 1000],
                       [self.path[i][path_len - 1][2] / 1000], color='green', alpha=0.7, label=f"UAV {i + 1}")
            # for j in range(path_len - 1):
            #     ax.scatter([self.path[i][j][0] / 1000], [self.path[i][j][1] / 1000],
            #                [self.path[i][j][2] / 1000], color='green', alpha=0.3)

            ax.plot3D(x, y, z, color='green')
            ax.plot3D([self.uav_init_state[i][0] / 1000, self.target[i][0] / 1000],
                      [self.uav_init_state[i][1] / 1000, self.target[i][1] / 1000],
                      [self.uav_init_state[i][2] / 1000, self.target[i][2] / 1000],
                      color='green', alpha=0.3, linestyle=':')

        ax.set_title("3D path")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_zlabel("z (km)")
        ax.set_xlim(0, 5)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_ylim(0, 5)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_zlim(0, 0.3)
        ax.zaxis.set_major_locator(MultipleLocator(0.05))
        ax.legend()
        plt.show()

    def _show_xy_path(self):

        for i in range(self.n_uav):
            path_len = len(self.path[i])
            x = [self.path[i][j][0] / 1000 for j in range(path_len)]
            y = [self.path[i][j][1] / 1000 for j in range(path_len)]
            plt.scatter([self.path[i][path_len - 1][0] / 1000], [self.path[i][path_len - 1][1] / 1000],
                        color='green', alpha=0.7, label=f"UAV {i + 1}")
            for j in range(path_len - 1):
                plt.scatter([self.path[i][j][0] / 1000], [self.path[i][j][1] / 1000],
                            color='green', alpha=0.3, marker=self.marker_list[i])
            plt.plot(x, y, color='green')
            plt.plot([self.uav_init_state[i][0] / 1000, self.target[i][0] / 1000],
                     [self.uav_init_state[i][1] / 1000, self.target[i][1] / 1000],
                     color='green', alpha=0.3, linestyle=':')

            circle1 = plt.Circle(xy=([self.path[i][path_len - 1][0] / 1000], [self.path[i][path_len - 1][1] / 1000]),
                                 radius=self.min_sep_hori / 1000, linestyle=':', color='green', fill=False)
            plt.gca().add_patch(circle1)
            circle2 = plt.Circle(xy=(2500 / 1000, 2500 / 1000), radius=2500 / 1000, linestyle=':', color='black',
                                 fill=False)
            plt.gca().add_patch(circle2)

        plt.axis('equal')
        plt.title("2D path - xy")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.legend()
        plt.show()

    def _show_xz_path(self):
        for i in range(self.n_uav):
            path_len = len(self.path[i])
            x = [self.path[i][j][0] / 1000 for j in range(path_len)]
            z = [self.path[i][j][2] / 1000 for j in range(path_len)]
            plt.scatter([self.path[i][path_len - 1][0] / 1000], [self.path[i][path_len - 1][2] / 1000],
                        color='green', alpha=0.7, label=f"UAV {i + 1}")
            # for j in range(path_len - 1):
            #     plt.scatter([self.path[i][j][0] / 1000], [self.path[i][j][2] / 1000],
            #                 color='green', alpha=0.3)
            plt.plot(x, z, color='green')
            plt.plot([self.uav_init_state[i][0] / 1000, self.target[i][0] / 1000],
                     [self.uav_init_state[i][2] / 1000, self.target[i][2] / 1000],
                     color='green', alpha=0.3, linestyle=':')

            rectangle1 = plt.Rectangle(xy=(self.path[i][path_len - 1][0] / 1000 - self.min_sep_hori / 1000,
                                           self.path[i][path_len - 1][2] / 1000 - self.min_sep_vert / 1000),
                                       width=self.min_sep_hori * 2 / 1000,
                                       height=self.min_sep_vert * 2 / 1000,
                                       color='green', linestyle=':', fill=False)
            plt.gca().add_patch(rectangle1)

        plt.title("2D path - xz")
        plt.xlabel("x (km)")
        plt.ylabel("z (km)")
        plt.xlim(0, 5)
        plt.ylim(0, 0.3)
        plt.legend()
        plt.show()


class Viewer(pyglet.window.Window):
    def __init__(self, uav_state, target):
        super(Viewer, self).__init__(width=500, height=500, resizable=False, caption='single-agent',
                                     vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.uav_state = uav_state
        self.uav = []
        for i in range(len(self.uav_state)):
            cur_uav = pyglet.shapes.Circle(self.uav_state[i][0] / 10, self.uav_state[i][1] / 10, 3,
                                           color=(1, 100, 1))
            self.uav.append(cur_uav)
        self.target = target
        self.tar = []
        for i in range(len(self.uav_state)):
            cur_tar = pyglet.shapes.Circle(self.target[i][0] / 10, self.target[i][1] / 10, 3, color=(1, 200, 1))
            self.tar.append(cur_tar)

    def render(self):
        self._update_uav()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        for uav in self.uav:
            uav.draw()
        for tar in self.tar:
            tar.draw()

    def _update_uav(self):
        temp_uav = []
        for i in range(len(self.uav)):
            temp_uav.append(pyglet.shapes.Circle(self.uav_state[i][0] / 10, self.uav_state[i][1] / 10, 3,
                                                 color=(1, 100, 1)))
        self.uav = temp_uav


if __name__ == "__main__":
    env = Env(mode="train")
    init_state = env.reset()

    rewards = 0.0
    for i in range(50):
        env.render()
        time.sleep(0.1)
        action = env.sample_action()
        s, r, done, _ = env.step(action)
        print(f"currently, the {i + 1} step:\n"
              # f"           {s[0][15]}\n"
              f"           Action: speed {action[0][0] * 5.0, action[0][1] * 5.0, action[0][2] * 0.5}\n"
              f"           State: pos {s[0][0] * 5000.0, s[0][1] * 5000.0, s[0][2] * 300.0};   speed {s[0][3] * 200, s[0][4] * 200.0, s[0][5] * 10}\n"
              f"           Reward:{r[0]}\n"
              # f"           {s[1][15]}\n"
              f"           Action: speed {action[1][0] * 5.0, action[1][1] * 5.0, action[1][2] * 0.5}\n"
              f"           State: pos {s[1][0] * 5000.0, s[1][1] * 5000.0, s[1][2] * 300.0};   speed {s[1][3] * 200, s[1][4] * 200.0, s[1][5] * 10}\n"
              f"           Reward:{r[1]}\n")
        rewards += sum(r)
        if np.all(done):
            break
    env.show_path()
    print(rewards)
    env.close()
