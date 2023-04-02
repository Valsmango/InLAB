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
state设定：【x，y，z，v_x，v_y，v_z，tar_x，tar_y，tar_z，obs_x，obs_y，obs_z】

[{'x': 0, 'y': 0, 'z': 0, 'v_x': 100, 'v_y': 100, 'v_z': 6, 'tar_x': 5000, 'tar_y': 5000, 'tar_z': 300, 'obs_x': 0, 'obs_y': 0, 'obs_z': 0}]
[{'x': 0, 'y': 0, 'z': 0, 'v_hori': 100 * np.sqrt(2), 'v_vert': 6, 'angle_hori': (1 / 4) * np.pi}]

action设定：【dvx， dvy， dvz】
'''

class Env(object):
    def __init__(self):
        self.uav_state = []
        self.path = []
        self.delta_t = 1
        self.min_sep_hori = 152.4
        self.min_sep_vert = 30.48
        self.min_range = 1000
        self.viewer = None
        self._max_episode_steps = 200

        self.max_action = np.array([10.0, 10.0, 2.0])
        # ndarray - tuple
        self.static_obs_state = []
        self.dynamic_obs_state = []
        self.uav_init_state = self.uav_state
        self.dynamic_obs_init_state = self.dynamic_obs_state

        # 随机起点和终点
        x = np.random.rand() * 300  # 避免一步就无了
        y = np.random.rand() * 5000
        z = np.random.rand() * 200 + 50
        tar_x = np.random.rand() * 300 + 4700
        tar_y = np.random.rand() * 5000
        tar_z = np.random.rand() * 200 + 50

        init_start = [[x, y, z, (tar_x - x) / 50, (tar_y - y) / 50, (tar_z - z) / 50, tar_x, tar_y, tar_z, 0, 0, 0]]
        init_target = [[tar_x, tar_y, tar_z]]

        self._init_map(init_start, init_target)

    def reset(self):
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        self.path = []
        self.path.append([self.uav_state[0][0], self.uav_state[0][1], self.uav_state[0][2]])
        # 【x，y，z，v_x，v_y，v_z，tar_x，tar_y，tar_z，obs_x，obs_y，obs_z】
        return_state = np.array(self.uav_state[0]) / np.array([5000.0, 5000.0, 300.0,
                                                            200.0, 200.0, 10.0,
                                                            5000.0, 5000.0, 300.0,
                                                            5000.0, 5000.0, 300.0])
        return torch.Tensor(return_state)

    def step(self, input_action):
        action = np.array(input_action) * self.max_action
        # action noise
        action += np.random.normal(loc=0, scale=0.003, size=len(self.max_action))
        done = False
        reward = 0.0
        category = 0  # 0表示普通的终止，1表示成功，2表示失败

        # Calculate the uav's position
        self.uav_state[0][3] += action[0]
        if self.uav_state[0][3] > 200:
            self.uav_state[0][3] = 200
            reward += -1
        elif self.uav_state[0][3] < -200:
            self.uav_state[0][3] = -200
            reward += -1
        self.uav_state[0][0] += self.uav_state[0][3] * self.delta_t
        self.uav_state[0][4] += action[1]
        if self.uav_state[0][4] > 200:
            self.uav_state[0][4] = 200
            reward += -1
        elif self.uav_state[0][4] < -200:
            self.uav_state[0][4] = -200
            reward += -1
        self.uav_state[0][1] += self.uav_state[0][4] * self.delta_t
        self.uav_state[0][5] += action[2]
        if self.uav_state[0][5] > 10:
            self.uav_state[0][5] = 10
            reward += -1
        elif self.uav_state[0][5] < -10:
            self.uav_state[0][5] = -10
            reward += -1
        self.uav_state[0][2] += self.uav_state[0][5] * self.delta_t
        self.path.append([self.uav_state[0][0], self.uav_state[0][1], self.uav_state[0][2]])
        # Calculate the dynamic obstacles' position
        for obs in self.dynamic_obs_state:
            obs['x'] += obs['v_x'] * self.delta_t
            obs['y'] += obs['v_y'] * self.delta_t
            obs['z'] += obs['v_z'] * self.delta_t
        # Calculate the reward
        static_obs_dis = [np.sqrt(
            (self.static_obs_state[i]['x'] - self.uav_state[0][0]) ** 2 +
            (self.static_obs_state[i]['y'] - self.uav_state[0][1]) ** 2 +
            (self.static_obs_state[i]['z'] - self.uav_state[0][2]) ** 2) for i in range(len(self.static_obs_state))]
        static_obs_dis_hori = [np.sqrt(
            (self.static_obs_state[i]['x'] - self.uav_state[0][0]) ** 2 +
            (self.static_obs_state[i]['y'] - self.uav_state[0][1]) ** 2) for i in range(len(self.static_obs_state))]
        static_obs_dis_vert = [np.abs(self.static_obs_state[i]['z'] - self.uav_state[0][2]) for i in
                               range(len(self.static_obs_state))]
        dynamic_obs_dis = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0][0]) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0][1]) ** 2 +
            (self.dynamic_obs_state[i]['z'] - self.uav_state[0][2]) ** 2) for i in range(len(self.dynamic_obs_state))]
        dynamic_obs_dis_hori = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0][0]) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0][1]) ** 2) for i in range(len(self.dynamic_obs_state))]
        dynamic_obs_dis_vert = [np.abs(self.dynamic_obs_state[i]['z'] - self.uav_state[0][2]) for i in
                                range(len(self.dynamic_obs_state))]
        tar_dis = np.sqrt((self.uav_state[0][0] - self.target[0][0]) ** 2 +
                          (self.uav_state[0][1] - self.target[0][1]) ** 2 +
                          (self.uav_state[0][2] - self.target[0][2]) ** 2)
        tar_dis_hori = np.sqrt((self.uav_state[0][0] - self.target[0][0]) ** 2 +
                               (self.uav_state[0][1] - self.target[0][1]) ** 2)
        tar_dis_vert = np.abs(self.uav_state[0][2] - self.target[0][2])
        if min(static_obs_dis) > min(dynamic_obs_dis):
            nearest_obs_idx = dynamic_obs_dis.index(min(dynamic_obs_dis))
            self.uav_state[0][9] = self.dynamic_obs_state[nearest_obs_idx]['x']
            self.uav_state[0][10] = self.dynamic_obs_state[nearest_obs_idx]['y']
            self.uav_state[0][11] = self.dynamic_obs_state[nearest_obs_idx]['z']
        else:
            nearest_obs_idx = static_obs_dis.index(min(static_obs_dis))
            self.uav_state[0][9] = self.static_obs_state[nearest_obs_idx]['x']
            self.uav_state[0][10] = self.static_obs_state[nearest_obs_idx]['y']
            self.uav_state[0][11] = self.static_obs_state[nearest_obs_idx]['z']

        urep_static = np.where(np.array(static_obs_dis) < self.min_range,
                               - 1 / 2 * self.rep * (1 / np.array(static_obs_dis) - 1 / self.min_range), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
                                - 1 / 2 * self.rep * (1 / np.array(dynamic_obs_dis) - 1 / self.min_range), 0)
        uatt = - self.att * tar_dis
        reward += 0.00125 * sum(urep_static) + 0.00125 * sum(urep_dynamic)
        reward += 0.0025 * uatt

        reward += -0.5        # 这一信息让其快速到达终点
        # reward += (pre_tar_dis - tar_dis)/10

        if tar_dis_vert < self.min_sep_vert and tar_dis_hori < self.min_sep_hori:
            reward += 100
            category = 1
            done = True
        elif np.any(np.array(np.array(static_obs_dis_vert) < self.min_sep_vert) &
                    np.array(np.array(static_obs_dis_hori) < self.min_sep_hori)) or \
                np.any(np.array(np.array(dynamic_obs_dis_vert) < self.min_sep_vert) &
                       np.array(np.array(dynamic_obs_dis_hori) < self.min_sep_hori)):
            reward += -100
            category = 2
            done = True
        elif self.uav_state[0][0] < 0.0 or self.uav_state[0][0] > 5000.0 or \
                self.uav_state[0][1] < 0.0 or self.uav_state[0][1] > 5000.0 or \
                self.uav_state[0][2] < 0.0 or self.uav_state[0][2] > 300.0:
            reward += -100
            done = True
        elif len(self.path) > self._max_episode_steps:
            done = True

        return_state = np.array(self.uav_state[0]) / np.array([5000.0, 5000.0, 300.0,
                                             200.0, 200.0, 10.0,
                                             5000.0, 5000.0, 300.0,
                                             5000.0, 5000.0, 300.0])
        return torch.Tensor(return_state), reward, done, category

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.uav_state, self.target, self.static_obs_state, self.dynamic_obs_state)
        self.viewer.render()

    def sample_action(self):
        # # Mean
        random_delta_v_x = np.random.rand() * 2 - 1
        random_delta_v_y = np.random.rand() * 2 - 1
        random_delta_v_z = np.random.rand() * 2 - 1
        return torch.Tensor([random_delta_v_x, random_delta_v_y, random_delta_v_z])

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

    def plot_reward(self, x, y, z=150):
        reward = 0.0
        # static_obs_dis = [np.sqrt(
        #     (self.static_obs_state[i]['x'] - x) ** 2 +
        #     (self.static_obs_state[i]['y'] - y) ** 2 +
        #     (self.static_obs_state[i]['z'] - z) ** 2) for i in range(len(self.static_obs_state))]
        # dynamic_obs_dis = [np.sqrt(
        #     (self.dynamic_obs_state[i]['x'] - x) ** 2 +
        #     (self.dynamic_obs_state[i]['y'] - y) ** 2 +
        #     (self.dynamic_obs_state[i]['z'] - z) ** 2) for i in range(len(self.dynamic_obs_state))]
        # tar_dis = np.sqrt((x - self.target[0][0]) ** 2 +
        #                   (y - self.target[0][1]) ** 2 +
        #                   (z - self.target[0][2]) ** 2)
        #
        # urep_static = np.where(np.array(static_obs_dis) < self.min_range,
        #                        - 1 / 2 * self.rep * (1 / np.array(static_obs_dis) - 1 / self.min_range), 0)
        # urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
        #                         - 1 / 2 * self.rep * (1 / np.array(dynamic_obs_dis) - 1 / self.min_range), 0)
        # uatt = - self.att * tar_dis
        # reward += 0.0025*sum(urep_static) + 0.0025*sum(urep_dynamic)
        # reward += 0.0025 * uatt

        static_obs_dis_hori = [np.sqrt(
            (self.static_obs_state[i]['x'] - x) ** 2 +
            (self.static_obs_state[i]['y'] - y) ** 2) for i in range(len(self.static_obs_state))]
        static_obs_dis_vert = [np.abs(self.static_obs_state[i]['z'] - z) for i in range(len(self.static_obs_state))]
        dynamic_obs_dis_hori = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - x) ** 2 +
            (self.dynamic_obs_state[i]['y'] - y) ** 2) for i in range(len(self.dynamic_obs_state))]
        dynamic_obs_dis_vert = [np.abs(self.dynamic_obs_state[i]['z'] - z) for i in range(len(self.dynamic_obs_state))]
        urep_static = np.where(np.array(static_obs_dis_hori) < self.min_range,
                               - np.exp(- np.array(static_obs_dis_hori) / 100), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis_hori) < self.min_range,
                                - np.exp(- np.array(dynamic_obs_dis_hori) / 100), 0)
        reward += 1*(sum(urep_static) + sum(urep_dynamic))
        tar_dis_hori = np.sqrt((x - self.target[0][0]) ** 2 + (y - self.target[0][1]) ** 2)
        tar_dis_vert = np.abs(z - self.target[0][2])
        uatt = np.exp(tar_dis_hori / 2000)
        # reward += uatt
        return reward

    def plot_env(self):
        ax = plt.axes(projection='3d')

        static_obs_len = len(self.static_obs_state)
        static_xscatter = [self.static_obs_state[i]['x'] / 1000 for i in range(static_obs_len)]
        static_yscatter = [self.static_obs_state[i]['y'] / 1000 for i in range(static_obs_len)]
        static_zscatter = [self.static_obs_state[i]['z'] / 1000 for i in range(static_obs_len)]
        ax.scatter(static_xscatter, static_yscatter, static_zscatter, label='static obstacle', c='black', alpha=0.7)

        dynamic_obs_len = len(self.dynamic_obs_init_state)
        dynamic_xscatter = [self.dynamic_obs_init_state[i]['x'] / 1000 for i in range(dynamic_obs_len)]
        dynamic_yscatter = [self.dynamic_obs_init_state[i]['y'] / 1000 for i in range(dynamic_obs_len)]
        dynamic_zscatter = [self.dynamic_obs_init_state[i]['z'] / 1000 for i in range(dynamic_obs_len)]
        # ax.scatter(dynamic_xscatter, dynamic_yscatter, dynamic_zscatter, c='r', alpha=0.3)

        dynamic_obs_cur_len = len(self.dynamic_obs_state)
        dynamic_xscatter_cur = [self.dynamic_obs_state[i]['x'] / 1000 for i in range(dynamic_obs_cur_len)]
        dynamic_yscatter_cur = [self.dynamic_obs_state[i]['y'] / 1000 for i in range(dynamic_obs_cur_len)]
        dynamic_zscatter_cur = [self.dynamic_obs_state[i]['z'] / 1000 for i in range(dynamic_obs_cur_len)]
        ax.scatter(dynamic_xscatter_cur, dynamic_yscatter_cur, dynamic_zscatter_cur, label='dynamic obstacle',
                   c='r', alpha=0.7)

        for k in range(dynamic_obs_len):
            ax.plot3D([dynamic_xscatter_cur[k], dynamic_xscatter[k]],
                      [dynamic_yscatter_cur[k], dynamic_yscatter[k]],
                      [dynamic_zscatter_cur[k], dynamic_zscatter[k]],
                      alpha=0.3, c='r', linestyle=':')

        # ax.set_title("Scene 1")
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

    def _init_map(self, init_start, init_target):
        self.target = init_target
        self.uav_init_state = init_start

        # #################################### Scene 1 : 40 static + 10 dynamic #################################
        # self.static_obs_state = [{'x': 1381.1731790938877, 'y': 4289.2199856158695, 'z': 24.50305664412391, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2931.605624788499, 'y': 4445.080836900117, 'z': 77.99355622245042, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3236.5627193197274, 'y': 2878.3406416516696, 'z': 178.1545316580548, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1343.9772744060385, 'y': 1079.3929407306493, 'z': 86.08722980763662, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1674.371387024943, 'y': 2402.8866607353893, 'z': 250.1925288002626, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4205.244319721307, 'y': 1540.2288974972178, 'z': 162.79039134395006, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3642.8131052894155, 'y': 913.2952524822091, 'z': 130.80948507387137, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1724.9053970703337, 'y': 2788.80921826057, 'z': 290.1193952542223, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 885.2143191134645, 'y': 1651.0125317030368, 'z': 2.6811351108014914, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2382.869881278147, 'y': 3080.7344370142273, 'z': 285.6969344746394, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3131.0481375864347, 'y': 893.0263528506473, 'z': 240.52050033462345, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4249.904927289404, 'y': 3333.2060503334574, 'z': 41.846506376985474, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 850.5117012897388, 'y': 3661.3684987000715, 'z': 185.49395430575333, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 631.761071950037, 'y': 2171.6068933878423, 'z': 34.1951905640148, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1634.1019512341156, 'y': 3441.3333395217105, 'z': 163.99505336015952, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2777.8816451873095, 'y': 648.2267321686601, 'z': 191.78883720918506, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3713.517342112914, 'y': 4323.988532879086, 'z': 122.58047407488753, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3347.474865597186, 'y': 874.1175931881928, 'z': 107.83373349308657, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 976.7999881277909, 'y': 3880.623651646794, 'z': 66.9066777828683, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1299.4579551446752, 'y': 705.614418043246, 'z': 254.59821967333758, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2795.701246365724, 'y': 2316.057078152472, 'z': 125.06356688115935, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1233.431017176224, 'y': 4271.425005605422, 'z': 184.99680706580415, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 910.4427300608218, 'y': 4233.224223256477, 'z': 240.1299738714734, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 646.964847446597, 'y': 4432.7398689645415, 'z': 165.10429710621483, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1567.98673423435, 'y': 3030.11509665907, 'z': 10.370909807814211, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2706.602816734253, 'y': 1499.4318243634436, 'z': 93.6038773708194, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 661.0257323531754, 'y': 4012.173227369368, 'z': 32.67268158500728, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 937.5873841570748, 'y': 2188.2196531090426, 'z': 63.996732423958406, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1315.2324576422848, 'y': 627.5541323933166, 'z': 248.58814562135046, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3022.947857896785, 'y': 1534.4971386802079, 'z': 84.2429949283505, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3110.855980078808, 'y': 654.9801356600873, 'z': 157.205551753061, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2916.447908849917, 'y': 3480.9451312289166, 'z': 297.1273630991828, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3371.8017192186544, 'y': 4176.102879508455, 'z': 65.70472990331008, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2958.2085938024015, 'y': 4287.894148747394, 'z': 236.90907984596774, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3478.6981122207144, 'y': 1061.9170516534098, 'z': 4.017868642110944, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3387.729542385443, 'y': 2291.45110453249, 'z': 296.5917404524909, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1116.6644618777384, 'y': 2879.7170887453894, 'z': 255.93727192674942, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1455.8268701735678, 'y': 4234.062984406286, 'z': 225.58025624355142, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2002.154082134516, 'y': 4218.176874576926, 'z': 12.655640106422606, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2803.9695714974205, 'y': 1535.1750030065464, 'z': 291.46676254151066, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}]
        # self.dynamic_obs_init_state = [{'x': 4147.21087148082, 'y': 3599.9936007699644, 'z': 103.70666090387277, 'v_x': 15.505108872277013, 'v_y': 11.590027930831843, 'v_z': 0.5758937277213043}, {'x': 3688.114673757661, 'y': 1012.8671055540108, 'z': 246.60557800307538, 'v_x': 23.815528024572423, 'v_y': 22.343273599578385, 'v_z': -0.6274841892864574}, {'x': 529.565267741602, 'y': 3944.233591216235, 'z': 110.52228329101834, 'v_x': 13.878936745912407, 'v_y': 17.568271779486956, 'v_z': 0.4203348927323103}, {'x': 1086.5730904702007, 'y': 1875.1358542513242, 'z': 5.37079865314446, 'v_x': 8.116336332062266, 'v_y': 23.018339088663044, 'v_z': 1.2525202862541973}, {'x': 1073.3757854278654, 'y': 2058.8169498960324, 'z': 271.0228812029455, 'v_x': 16.42093967863713, 'v_y': 17.466943030105988, 'v_z': 0.44161427968595923}, {'x': 4164.538570407007, 'y': 1682.2451374008112, 'z': 120.83318868489215, 'v_x': 24.8020795658355, 'v_y': 10.145459424036959, 'v_z': -1.126628058012567}, {'x': 4436.002836873984, 'y': 1813.647891060208, 'z': 135.47275398366986, 'v_x': 6.8693712477334685, 'v_y': 4.687205210115761, 'v_z': 0.7415848965097491}, {'x': 2210.4191702943763, 'y': 4199.496776429098, 'z': 196.3618293569498, 'v_x': 15.373646140378833, 'v_y': 24.717935560445216, 'v_z': -0.5129133662024087}, {'x': 4016.970574532727, 'y': 1239.985435516733, 'z': 144.18472541967603, 'v_x': 23.350173785680575, 'v_y': 15.6613928897861, 'v_z': -1.2335985572652928}, {'x': 1880.8181189214824, 'y': 3264.5014400132955, 'z': 139.1761621981652, 'v_x': 20.87831584319002, 'v_y': 5.057161671402283, 'v_z': -1.1531940900115072}]

        #################################### Scene 2 : 10 static + 40 dynamic #################################
        self.static_obs_state = [{'x': 3114.8036405212574, 'y': 4207.761865057853, 'z': 294.188671514354, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2906.222739408599, 'y': 870.6145734645316, 'z': 85.94123111158397, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4408.631358319608, 'y': 1127.27572144982, 'z': 75.54072405174173, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 581.3455279529607, 'y': 651.5662696459779, 'z': 73.96841514902093, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3535.6708387892295, 'y': 1241.9965849711562, 'z': 228.62140476213108, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1033.6688426457558, 'y': 1229.7977966766844, 'z': 199.41048999302637, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2662.013046793342, 'y': 3714.2845298353363, 'z': 169.013482913838, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3124.2499632898175, 'y': 984.4406836824024, 'z': 90.28956406662515, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1294.3171444862824, 'y': 1551.5224874464125, 'z': 55.25459602413505, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3736.139804951551, 'y': 2798.745709213755, 'z': 142.63800894895047, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}]
        self.dynamic_obs_init_state = [{'x': 1053.7001688804758, 'y': 1161.5012616804586, 'z': 120.83373095314778, 'v_x': 4.86605925867779, 'v_y': 3.5801740591896447, 'v_z': 1.165422621867112}, {'x': 4115.838229503943, 'y': 2859.687817468106, 'z': 295.2873765387844, 'v_x': 21.261817351950427, 'v_y': -22.153377559744637, 'v_z': 0.36872088962168315}, {'x': 768.6471402062591, 'y': 1828.4722030603043, 'z': 96.40687250577035, 'v_x': 1.9108718219016367, 'v_y': 15.12184911527185, 'v_z': -0.10985878863968823}, {'x': 3009.5171564991406, 'y': 1936.8793551279286, 'z': 71.53922928706655, 'v_x': -21.682206453372025, 'v_y': 6.591138779320875, 'v_z': -0.22678078599084128}, {'x': 1501.7676205760986, 'y': 2973.078932000869, 'z': 163.35244804751417, 'v_x': 20.263711974655187, 'v_y': -0.26106899168324915, 'v_z': 0.29098897026608506}, {'x': 3635.231206419609, 'y': 1993.06099308573, 'z': 264.1973032976388, 'v_x': -11.710628359535525, 'v_y': -22.922998164193793, 'v_z': -0.8251758963245691}, {'x': 1407.6957585441119, 'y': 1203.1952104459392, 'z': 51.37587593838726, 'v_x': 3.9120755305492487, 'v_y': -20.250959181001917, 'v_z': -0.19157147902056693}, {'x': 1490.1766829639098, 'y': 3424.261711327554, 'z': 45.87332031213928, 'v_x': -4.929074156597867, 'v_y': 7.023726119376114, 'v_z': 0.30103141056412785}, {'x': 2205.867511324105, 'y': 3654.515347813332, 'z': 35.08506355571822, 'v_x': 6.685037503566139, 'v_y': -16.376200093978106, 'v_z': 1.175406276833722}, {'x': 2934.898017472873, 'y': 1491.7200226748687, 'z': 258.5991993882242, 'v_x': 5.758604762130648, 'v_y': 12.316869881463425, 'v_z': 1.0505814798462225}, {'x': 845.3238764268449, 'y': 3011.9453971543467, 'z': 240.83614374933137, 'v_x': -9.624595464342356, 'v_y': 1.5402875177805093, 'v_z': 0.8955737889975537}, {'x': 3108.527817543942, 'y': 3847.3906872508614, 'z': 86.21802587188355, 'v_x': -8.822632952543596, 'v_y': -19.028371810381355, 'v_z': 0.9496183464823083}, {'x': 1716.320310332686, 'y': 815.9066177790817, 'z': 161.53000264645374, 'v_x': -18.115589406722975, 'v_y': 20.03999043076893, 'v_z': -0.025462740353988433}, {'x': 1536.8762396608026, 'y': 2915.3034922834127, 'z': 96.8311181286028, 'v_x': 8.94762209524928, 'v_y': 16.124007995861902, 'v_z': -0.8780142584617452}, {'x': 3206.706716909681, 'y': 912.2440103622238, 'z': 106.29559184323777, 'v_x': -4.944422640154716, 'v_y': 24.01794086803462, 'v_z': 0.8726373152332974}, {'x': 3241.920638565184, 'y': 3639.288300294529, 'z': 80.56095115666221, 'v_x': -13.629692293227357, 'v_y': -19.65557438328259, 'v_z': -0.31019218784930525}, {'x': 3725.3777779880934, 'y': 3398.345627417584, 'z': 270.25759802167875, 'v_x': -0.9347285172344897, 'v_y': -5.292476588567769, 'v_z': 1.20068686172687}, {'x': 1627.140435916719, 'y': 4459.487346321804, 'z': 298.499465103248, 'v_x': -15.24484653878837, 'v_y': -3.020615736688004, 'v_z': -0.834338933946256}, {'x': 569.3564152404575, 'y': 2982.1468565397327, 'z': 108.8279665899177, 'v_x': -4.821116502827426, 'v_y': -4.286301725014386, 'v_z': -0.5531426385395595}, {'x': 3045.059637139595, 'y': 2910.8454422262157, 'z': 202.71126908118447, 'v_x': -16.856540719033802, 'v_y': -0.3983225067320362, 'v_z': -0.8614695968291248}, {'x': 4138.777371373904, 'y': 2503.7957876159003, 'z': 145.3110289661744, 'v_x': 6.45122449239452, 'v_y': -20.17425382377676, 'v_z': -0.34753640518997564}, {'x': 723.0074339578971, 'y': 1393.3026353447476, 'z': 90.3914346048176, 'v_x': 14.60541696888297, 'v_y': 10.035386426011748, 'v_z': 0.7318002294306991}, {'x': 2138.8832139748906, 'y': 2243.209069965221, 'z': 161.74450404062202, 'v_x': 8.560299191967978, 'v_y': 13.493907103444819, 'v_z': 0.5907009230753992}, {'x': 4043.1324124870875, 'y': 3973.658120591518, 'z': 288.41563827573464, 'v_x': -3.50979157672316, 'v_y': -12.243258000912338, 'v_z': -0.18032609848942416}, {'x': 3313.0937906130443, 'y': 1776.6569847197084, 'z': 148.66022162567836, 'v_x': 3.985057250857256, 'v_y': 17.25550256494016, 'v_z': 0.748603617011983}, {'x': 2088.9833628088727, 'y': 2880.4552080629974, 'z': 247.73183237035548, 'v_x': 2.313572218737942, 'v_y': 2.53170401248013, 'v_z': 0.08540778894067191}, {'x': 2793.6966221814196, 'y': 3832.5693621026126, 'z': 266.0751860324688, 'v_x': 10.146656314133388, 'v_y': 4.900257295678053, 'v_z': -0.8218218855130762}, {'x': 3220.2699632859776, 'y': 4201.201506192434, 'z': 250.07910909772912, 'v_x': 22.731099368556656, 'v_y': 19.781866004919934, 'v_z': -0.3254621782408753}, {'x': 3171.0270236901406, 'y': 1115.0683059177632, 'z': 73.24368403267162, 'v_x': 10.410899204502343, 'v_y': -9.760594652477595, 'v_z': -0.49553690152629537}, {'x': 2004.6061686489156, 'y': 917.2951048600919, 'z': 127.34655576237459, 'v_x': 23.586845353915045, 'v_y': 12.074245263959646, 'v_z': 0.3431346380020144}, {'x': 1679.9941699157496, 'y': 3211.417724491626, 'z': 4.635473940381396, 'v_x': 1.950383693201939, 'v_y': 0.7223465840133017, 'v_z': 1.2114688565549696}, {'x': 1176.2333535677124, 'y': 2461.842350574712, 'z': 44.45366046805377, 'v_x': 17.113380256259894, 'v_y': -9.090437988044563, 'v_z': -0.8822831192538408}, {'x': 2986.074254743378, 'y': 4302.132896363493, 'z': 290.9225794704854, 'v_x': 5.470283868405062, 'v_y': 12.346134920111737, 'v_z': -0.588244404573541}, {'x': 2686.399934582118, 'y': 4138.4201482104445, 'z': 149.34328954147028, 'v_x': 3.8868837843408564, 'v_y': -17.91303413651901, 'v_z': 0.9110232963162703}, {'x': 2002.314578308547, 'y': 4368.97465910183, 'z': 243.56330256869833, 'v_x': 6.650090474135862, 'v_y': 4.440857615908676, 'v_z': -1.2331878669462617}, {'x': 2427.847473223252, 'y': 1605.0233449388666, 'z': 144.56487408342312, 'v_x': 8.35359915760663, 'v_y': -1.3316632902567846, 'v_z': -0.6072788761859748}, {'x': 3168.3070872978724, 'y': 2274.0167195231943, 'z': 28.09255036993974, 'v_x': 17.410438140291333, 'v_y': -8.003115529880052, 'v_z': -0.3110163644644408}, {'x': 3036.7030557092558, 'y': 1436.4178840018699, 'z': 281.8273036855491, 'v_x': 22.908593444982337, 'v_y': -12.09981433375229, 'v_z': -0.31796654635098665}, {'x': 4172.245284450668, 'y': 3223.270141554746, 'z': 194.75460098562965, 'v_x': -11.242395937450611, 'v_y': 21.465671101437444, 'v_z': -0.3967884763106069}, {'x': 4304.286101290847, 'y': 1490.9288054090068, 'z': 69.37576445322837, 'v_x': 17.045517534241164, 'v_y': 15.007728319501354, 'v_z': -0.04244433122776048}]

        # #################################### Scene 3 : 25 static + 25 dynamic #################################
        # self.static_obs_state =
        # self.dynamic_obs_init_state =

        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        self.att = 2 / ((self.target[0][0]/1000 - self.uav_init_state[0][0]/1000) ** 2 +
                        (self.target[0][1]/1000 - self.uav_init_state[0][1]/1000) ** 2 +
                        (self.target[0][2]/1000 - self.uav_init_state[0][2]/1000) ** 2)
        self.rep = 2 / (1 / self.min_sep_hori - 1 / self.min_range) ** 2
        # print(f"att parameter: {self.att}; rep parameter: {self.rep}")
        self.path.append([self.uav_init_state[0][0], self.uav_init_state[0][1], self.uav_init_state[0][2]])

        # update the nearest obs info
        static_obs_dis = [np.sqrt(
            (self.static_obs_state[i]['x'] - self.uav_state[0][0]) ** 2 +
            (self.static_obs_state[i]['y'] - self.uav_state[0][1]) ** 2 +
            (self.static_obs_state[i]['z'] - self.uav_state[0][2]) ** 2) for i in range(len(self.static_obs_state))]
        dynamic_obs_dis = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0][0]) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0][1]) ** 2 +
            (self.dynamic_obs_state[i]['z'] - self.uav_state[0][2]) ** 2) for i in range(len(self.dynamic_obs_state))]
        if min(static_obs_dis) > min(dynamic_obs_dis):
            nearest_obs_idx = dynamic_obs_dis.index(min(dynamic_obs_dis))
            self.uav_state[0][9] = self.dynamic_obs_state[nearest_obs_idx]['x']
            self.uav_state[0][10] = self.dynamic_obs_state[nearest_obs_idx]['y']
            self.uav_state[0][11] = self.dynamic_obs_state[nearest_obs_idx]['z']
            self.uav_init_state[0][9] = self.uav_state[0][9]
            self.uav_init_state[0][10] = self.uav_state[0][10]
            self.uav_init_state[0][11] = self.uav_state[0][11]
        else:
            nearest_obs_idx = static_obs_dis.index(min(static_obs_dis))
            self.uav_state[0][9] = self.static_obs_state[nearest_obs_idx]['x']
            self.uav_state[0][10] = self.static_obs_state[nearest_obs_idx]['y']
            self.uav_state[0][11] = self.static_obs_state[nearest_obs_idx]['z']
            self.uav_init_state[0][9] = self.uav_state[0][9]
            self.uav_init_state[0][10] = self.uav_state[0][10]
            self.uav_init_state[0][11] = self.uav_state[0][11]

    def _show_3D_path(self):
        ax = plt.axes(projection='3d')

        static_obs_len = len(self.static_obs_state)
        static_xscatter = [self.static_obs_state[i]['x'] / 1000 for i in range(static_obs_len)]
        static_yscatter = [self.static_obs_state[i]['y'] / 1000 for i in range(static_obs_len)]
        static_zscatter = [self.static_obs_state[i]['z'] / 1000 for i in range(static_obs_len)]
        ax.scatter(static_xscatter, static_yscatter, static_zscatter, label='static obstacle', c='black', alpha=0.7)

        dynamic_obs_len = len(self.dynamic_obs_init_state)
        dynamic_xscatter = [self.dynamic_obs_init_state[i]['x'] / 1000 for i in range(dynamic_obs_len)]
        dynamic_yscatter = [self.dynamic_obs_init_state[i]['y'] / 1000 for i in range(dynamic_obs_len)]
        dynamic_zscatter = [self.dynamic_obs_init_state[i]['z'] / 1000 for i in range(dynamic_obs_len)]
        # ax.scatter(dynamic_xscatter, dynamic_yscatter, dynamic_zscatter, c='r', alpha=0.3)

        dynamic_obs_cur_len = len(self.dynamic_obs_state)
        dynamic_xscatter_cur = [self.dynamic_obs_state[i]['x'] / 1000 for i in range(dynamic_obs_cur_len)]
        dynamic_yscatter_cur = [self.dynamic_obs_state[i]['y'] / 1000 for i in range(dynamic_obs_cur_len)]
        dynamic_zscatter_cur = [self.dynamic_obs_state[i]['z'] / 1000 for i in range(dynamic_obs_cur_len)]
        ax.scatter(dynamic_xscatter_cur, dynamic_yscatter_cur, dynamic_zscatter_cur, label='dynamic obstacle',
                   c='r', alpha=0.7)

        for k in range(dynamic_obs_len):
            ax.plot3D([dynamic_xscatter_cur[k], dynamic_xscatter[k]],
                      [dynamic_yscatter_cur[k], dynamic_yscatter[k]],
                      [dynamic_zscatter_cur[k], dynamic_zscatter[k]],
                      alpha=0.3, c='r', linestyle=':')

        path_len = len(self.path)
        x = [self.path[i][0] / 1000 for i in range(path_len)]
        y = [self.path[i][1] / 1000 for i in range(path_len)]
        z = [self.path[i][2] / 1000 for i in range(path_len)]
        ax.scatter([self.path[path_len - 1][0] / 1000], [self.path[path_len - 1][1] / 1000],
                   [self.path[path_len - 1][2] / 1000], color='green', alpha=0.7, label='UAV')
        ax.plot3D(x, y, z, color='green')
        for k in range(len(self.uav_init_state)):
            ax.plot3D([self.uav_init_state[k][0] / 1000, self.target[k][0] / 1000],
                      [self.uav_init_state[k][1] / 1000, self.target[k][1] / 1000],
                      [self.uav_init_state[k][2] / 1000, self.target[k][2] / 1000],
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
        static_obs_len = len(self.static_obs_state)
        static_xscatter = [self.static_obs_state[i]['x'] / 1000 for i in range(static_obs_len)]
        static_yscatter = [self.static_obs_state[i]['y'] / 1000 for i in range(static_obs_len)]
        plt.scatter(static_xscatter, static_yscatter, label='static obstacle', c='black', alpha=0.7)

        dynamic_obs_len = len(self.dynamic_obs_init_state)
        dynamic_xscatter = [self.dynamic_obs_init_state[i]['x'] / 1000 for i in range(dynamic_obs_len)]
        dynamic_yscatter = [self.dynamic_obs_init_state[i]['y'] / 1000 for i in range(dynamic_obs_len)]

        dynamic_obs_cur_len = len(self.dynamic_obs_state)
        dynamic_xscatter_cur = [self.dynamic_obs_state[i]['x'] / 1000 for i in range(dynamic_obs_cur_len)]
        dynamic_yscatter_cur = [self.dynamic_obs_state[i]['y'] / 1000 for i in range(dynamic_obs_cur_len)]
        plt.scatter(dynamic_xscatter_cur, dynamic_yscatter_cur, label='dynamic obstacle',
                    c='r', alpha=0.7)

        for k in range(dynamic_obs_len):
            plt.plot([dynamic_xscatter_cur[k], dynamic_xscatter[k]],
                     [dynamic_yscatter_cur[k], dynamic_yscatter[k]],
                     alpha=0.3, c='r', linestyle=':')

        path_len = len(self.path)
        x = [self.path[i][0] / 1000 for i in range(path_len)]
        y = [self.path[i][1] / 1000 for i in range(path_len)]
        plt.scatter([self.path[path_len - 1][0] / 1000], [self.path[path_len - 1][1] / 1000],
                    color='green', alpha=0.7, label='UAV')
        plt.plot(x, y, color='green')
        for k in range(len(self.uav_init_state)):
            plt.plot([self.uav_init_state[k][0] / 1000, self.target[k][0] / 1000],
                     [self.uav_init_state[k][1] / 1000, self.target[k][1] / 1000],
                     color='green', alpha=0.3, linestyle=':')

        circle = plt.Circle(xy=([self.path[path_len - 1][0] / 1000], [self.path[path_len - 1][1] / 1000]),
                            radius=self.min_sep_hori / 1000, linestyle=':', color='green', fill=False)
        plt.gca().add_patch(circle)

        # plt.axis('equal')
        plt.title("2D path - xy")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.legend()
        plt.show()

    def _show_xz_path(self):
        static_obs_len = len(self.static_obs_state)
        static_xscatter = [self.static_obs_state[i]['x'] / 1000 for i in range(static_obs_len)]
        static_zscatter = [self.static_obs_state[i]['z'] / 1000 for i in range(static_obs_len)]
        plt.scatter(static_xscatter, static_zscatter, label='static obstacle', c='black', alpha=0.7)

        dynamic_obs_len = len(self.dynamic_obs_init_state)
        dynamic_xscatter = [self.dynamic_obs_init_state[i]['x'] / 1000 for i in range(dynamic_obs_len)]
        dynamic_zscatter = [self.dynamic_obs_init_state[i]['z'] / 1000 for i in range(dynamic_obs_len)]

        dynamic_obs_cur_len = len(self.dynamic_obs_state)
        dynamic_xscatter_cur = [self.dynamic_obs_state[i]['x'] / 1000 for i in range(dynamic_obs_cur_len)]
        dynamic_zscatter_cur = [self.dynamic_obs_state[i]['z'] / 1000 for i in range(dynamic_obs_cur_len)]
        plt.scatter(dynamic_xscatter_cur, dynamic_zscatter_cur, label='dynamic obstacle',
                    c='r', alpha=0.7)

        for k in range(dynamic_obs_len):
            plt.plot([dynamic_xscatter_cur[k], dynamic_xscatter[k]],
                     [dynamic_zscatter_cur[k], dynamic_zscatter[k]],
                     alpha=0.3, c='r', linestyle=':')

        path_len = len(self.path)
        x = [self.path[i][0] / 1000 for i in range(path_len)]
        z = [self.path[i][2] / 1000 for i in range(path_len)]
        plt.scatter([self.path[path_len - 1][0] / 1000], [self.path[path_len - 1][2] / 1000],
                    color='green', alpha=0.7, label='UAV')
        plt.plot(x, z, color='green')
        for k in range(len(self.uav_init_state)):
            plt.plot([self.uav_init_state[k][0] / 1000, self.target[k][0] / 1000],
                     [self.uav_init_state[k][2] / 1000, self.target[k][2] / 1000],
                     color='green', alpha=0.3, linestyle=':')

        rectangle = plt.Rectangle(xy=(self.path[path_len - 1][0] / 1000 - self.min_sep_hori / 1000,
                                      self.path[path_len - 1][2] / 1000 - self.min_sep_vert / 1000),
                                  width=self.min_sep_hori * 2 / 1000,
                                  height=self.min_sep_vert * 2 / 1000,
                                  color='green', linestyle=':', fill=False)
        plt.gca().add_patch(rectangle)

        plt.title("2D path - xz")
        plt.xlabel("x (km)")
        plt.ylabel("z (km)")
        plt.xlim(0, 5)
        plt.ylim(0, 0.3)
        plt.legend()
        plt.show()


class Viewer(pyglet.window.Window):
    def __init__(self, uav_state, target, static_obs_state, dynamic_obs_state):
        super(Viewer, self).__init__(width=500, height=500, resizable=False, caption='single-agent',
                                     vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.uav_state = uav_state
        self.target = target
        self.uav = pyglet.shapes.Circle(self.uav_state[0][0] / 10, self.uav_state[0][1] / 10, 3,
                                        color=(1, 100, 1))
        self.target = pyglet.shapes.Circle(self.target[0][0] / 10, self.target[0][1] / 10, 3, color=(1, 200, 1))
        self.static_obs = []
        self.static_obs_state = static_obs_state
        for j in range(len(static_obs_state)):
            cur_obs = pyglet.shapes.Circle(self.static_obs_state[j]['x'] / 10, self.static_obs_state[j]['y'] / 10, 3,
                                           color=(86, 109, 249))
            self.static_obs.append(cur_obs)
        self.dynamic_obs = []
        self.dynamic_obs_state = dynamic_obs_state
        for j in range(len(dynamic_obs_state)):
            cur_obs = pyglet.shapes.Circle(self.dynamic_obs_state[j]['x'] / 10, self.dynamic_obs_state[j]['y'] / 10, 3,
                                           color=(249, 109, 249))
            self.dynamic_obs.append(cur_obs)

    def render(self):
        self._update_uav()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.uav.draw()
        self.target.draw()
        for obs in self.static_obs:
            obs.draw()
        for obs in self.dynamic_obs:
            obs.draw()

    def _update_uav(self):
        self.uav = pyglet.shapes.Circle(self.uav_state[0][0] / 10, self.uav_state[0][1] / 10, 3,
                                        color=(1, 100, 1))
        temp = []
        for j in range(len(self.dynamic_obs_state)):
            cur_obs = pyglet.shapes.Circle(self.dynamic_obs_state[j]['x'] / 10, self.dynamic_obs_state[j]['y'] / 10, 3,
                                           color=(249, 109, 249))
            temp.append(cur_obs)
        self.dynamic_obs = temp


if __name__ == "__main__":
    env = Env()
    init_state = env.reset()


    # for i in range(50):
    #     env.render()
    #     time.sleep(0.1)
    #     # action = env.sample_action()
    #     action = [0, 0, 0]
    #     s, r, done, _ = env.step(action)
    #     print(f"currently, the {i + 1} step:\n"
    #           f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
    #           f"           State: pos {s[0]*5000.0, s[1]*5000.0, s[2]*300.0};   speed {s[3]*200, s[4]*200.0, s[5]*10}\n"
    #           f"           Reward:{r}\n")
    #
    #     if done:
    #         env.show_path()
    #         break
    #
    # env.close()

    # ax = plt.axes(projection='3d')
    # x = np.zeros([500, 500])
    # y = np.zeros([500, 500])
    # r = np.zeros([500, 500])
    # for xx in range(500):
    #     for yy in range(500):
    #         x[xx][yy] = xx * 10
    #         y[xx][yy] = yy * 10
    #         r[xx][yy] = env.plot_reward(xx*10, yy*10, 200)
    # ax.scatter(x, y, r, s=0.01)
    # plt.show()

    # x = np.arange(-500, 5500, 10)
    # y = np.arange(-500, 5500, 10)
    # X, Y = np.meshgrid(x, y)
    # Z = env.plot_reward(X, Y)
    # ct = plt.contour(X, Y, Z, 100)
    # plt.clabel(ct, inline=True)
    # plt.show()

    print(len(env.dynamic_obs_state))
    print(len(env.static_obs_state))
    env.plot_env()
