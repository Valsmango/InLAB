# coding=utf-8
import time
import pyglet
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import copy

'''
state设定：【x，y，z，v_x，v_y，v_z，tar_x，tar_y，tar_z，obs_x，obs_y，obs_z】

[{'x': 0, 'y': 0, 'z': 0, 'v_x': 100, 'v_y': 100, 'v_z': 6, 'tar_x': 5000, 'tar_y': 5000, 'tar_z': 300, 'obs_x': 0, 'obs_y': 0, 'obs_z': 0}]
[{'x': 0, 'y': 0, 'z': 0, 'v_hori': 100 * np.sqrt(2), 'v_vert': 6, 'angle_hori': (1 / 4) * np.pi}]

action设定：【dvx， dvy， dvz】
'''


class SingleContinuousEnv(object):
    def __init__(self, seed=0):
        self.env_kinds = 4
        self._max_episode_steps = 100
        self.model = None
        self.randomly_choose_env()
        self.seed(seed)

    def randomly_choose_env(self):
        # randomly choose an env
        self.model = SingleContinuousEnv0(self._max_episode_steps,
                                          init_start=[{'x': 0, 'y': 0, 'z': 150,
                                                       'v_x': 100, 'v_y': 100, 'v_z': 0,
                                                       'tar_x': 5000, 'tar_y': 5000, 'tar_z': 150,
                                                       'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
                                          init_target=[{'x': 5000, 'y': 5000, 'z': 150}])
        # tmp = np.random.rand(1)
        # if tmp < 0.3:
        #     self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                       init_start=[{'x': 0, 'y': 0, 'z': 0,
        #                                                    'v_x': 100, 'v_y': 100, 'v_z': 6,
        #                                                    'tar_x': 5000, 'tar_y': 5000, 'tar_z': 300,
        #                                                    'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                       init_target=[{'x': 5000, 'y': 5000, 'z': 300}])
        # elif tmp < 0.6:
        #     self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                       init_start=[{'x': 0, 'y': 0, 'z': 0,
        #                                                    'v_x': 100, 'v_y': 100, 'v_z': 6,
        #                                                    'tar_x': 5000, 'tar_y': 5000, 'tar_z': 300,
        #                                                    'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                       init_target=[{'x': 5000, 'y': 5000, 'z': 300}])
        # else:
        #     self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                       init_start=[{'x': 0, 'y': 5000, 'z': 0,
        #                                                    'v_x': 100, 'v_y': 100, 'v_z': 6,
        #                                                    'tar_x': 5000, 'tar_y': 5000, 'tar_z': 300,
        #                                                    'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                       init_target=[{'x': 5000, 'y': 5000, 'z': 300}])

    def reset(self):
        return self.model.reset()

    def step(self, action):
        return self.model.step(action)

    def render(self):
        self.model.render()

    def sample_action(self):
        return self.model.sample_action()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def show_path(self):
        self.model.show_3D_path()
        self.model.show_2D_path()

    def close(self):
        self.model.close()

    def plot_reward(self, x, y, z):
        return self.model.plot_reward(x, y, z)


class SingleContinuousEnv0(object):
    def __init__(self, max_episode_steps, init_start, init_target):
        self.uav_state = []
        self.static_obstacle_num = 20
        self.dynamic_obstacle_num = 2
        # ndarray - tuple
        self.static_obs_state = []
        self.dynamic_obs_state = []
        self.uav_init_state = self.uav_state
        self.dynamic_obs_init_state = self.dynamic_obs_state
        self.path = []
        self.delta_t = 1
        self.min_sep = 100
        self.min_range = 1000
        self.viewer = None
        self._max_episode_steps = max_episode_steps

        self.init_map(init_start, init_target)

    def init_map(self, init_start, init_target):
        self.target = init_target
        self.uav_init_state = init_start
        self.static_obs_state = [{'x': 4003.793764727307, 'y': 853.1681619121373, 'z': 16.315866710215666, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2385.860678909147, 'y': 2769.7277838687073, 'z': 232.09502139960472, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1612.3007474232054, 'y': 2240.272161953309, 'z': 206.63276072578662, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4188.763754353382, 'y': 2761.6480122522826, 'z': 298.94622014598764, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 543.522074416406, 'y': 3943.2897372837033, 'z': 242.77316236805, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 18.405396704541666, 'y': 2103.5594566377213, 'z': 288.1453099557788, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4320.305967302095, 'y': 3301.515476250392, 'z': 190.0703240699427, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1092.3834790219184, 'y': 584.2479337109729, 'z': 201.77457580611534, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 870.7154906472664, 'y': 4209.422391283471, 'z': 234.21970685881348, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4137.580640017703, 'y': 4377.451353232047, 'z': 202.56578191076875, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2393.494787615238, 'y': 26.190496885956094, 'z': 217.85633327159317, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1000.6730563586108, 'y': 1337.4872318365094, 'z': 147.86314405539747, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 293.66578937926545, 'y': 1564.1679307627483, 'z': 105.65421164346466, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2080.1806466927874, 'y': 526.5314623173839, 'z': 140.6326346122918, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1356.9020889998762, 'y': 23.011551154897838, 'z': 224.61562585332365, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4951.826304084153, 'y': 1813.7366983945158, 'z': 123.15487382529727, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4732.987895688319, 'y': 2655.2854919978895, 'z': 104.22292627771142, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3031.6014012775945, 'y': 220.788010008931, 'z': 72.10868862696775, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1407.4879748268727, 'y': 18.450630482497175, 'z': 284.67019318574575, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2128.2475248749056, 'y': 1824.4898223439309, 'z': 277.02365016864206, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 848.9425767719538, 'y': 2740.3758863715543, 'z': 218.07458121062035, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1588.0651509830673, 'y': 2318.5793676475223, 'z': 27.945410637817602, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3377.8554388087146, 'y': 4549.22456158114, 'z': 113.95302605223586, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4690.5337918256855, 'y': 4860.283713679797, 'z': 275.51166940107885, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2163.8776321334667, 'y': 4544.18390726072, 'z': 229.9777090468076, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4053.7958549783452, 'y': 1689.8872611517074, 'z': 209.42690781801858, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 684.4556038736199, 'y': 4475.738916669855, 'z': 224.1242901174219, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2035.62869887899, 'y': 3909.4027647500084, 'z': 158.45917027199707, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2007.1529069657013, 'y': 4409.942410321135, 'z': 72.26486504538022, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 987.2212660478391, 'y': 2545.4289382466, 'z': 266.8008554248633, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1835.0903713064315, 'y': 4156.110071594966, 'z': 156.91788814038995, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 768.6109628064086, 'y': 4403.341641179685, 'z': 13.565536269356693, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3293.705951126722, 'y': 1602.2578978695067, 'z': 208.6943187436616, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4380.51625328133, 'y': 1119.182914217256, 'z': 10.893093025024104, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 294.7285322890575, 'y': 367.2056677392865, 'z': 236.38340419241695, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4178.398060225784, 'y': 4365.17729119024, 'z': 195.22205032076275, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2922.460610752454, 'y': 760.5947194989549, 'z': 237.6726174916348, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 515.1263438426695, 'y': 3596.04274248902, 'z': 66.25253767435206, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2539.8499377822013, 'y': 3344.6275326219416, 'z': 68.21033739578839, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2934.3229956146206, 'y': 4955.395592484076, 'z': 170.5425843044831, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4021.7819893115934, 'y': 4356.727705378222, 'z': 90.31298172612911, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3209.899125559299, 'y': 3543.644575623505, 'z': 296.4343911550105, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3176.140666075989, 'y': 2811.2327935240196, 'z': 246.12373728756114, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 597.9623171398097, 'y': 3022.162760285119, 'z': 222.58866025526225, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3474.2444413533303, 'y': 1879.1523164477935, 'z': 293.90427203684266, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4736.373625848054, 'y': 2081.4829719237223, 'z': 170.91781177427447, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 501.9327674886187, 'y': 2380.6813326475885, 'z': 250.04867664102477, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 770.4497134720057, 'y': 1536.6698589486232, 'z': 60.90215067956859, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1940.3836107518407, 'y': 2531.48946537218, 'z': 276.84399819721006, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3798.9612449084793, 'y': 929.2875670055728, 'z': 239.10644976224725, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}]
        self.dynamic_obs_init_state = [
            {'x': 2615.9051090096214, 'y': 1409.7206776842236, 'z': 88.66276900830434, 'v_x': 20.96222112806279,
             'v_y': 46.043242254770725, 'v_z': 1.2838647292989682},
            {'x': 1505.7763132271307, 'y': 2942.378573375284, 'z': 97.54520542993518, 'v_x': 44.816861352459135,
             'v_y': 31.602112005116044, 'v_z': -1.9816810285054403},
            {'x': 3071.2006836933365, 'y': 2879.8158733885625, 'z': 90.9175335595089, 'v_x': 1.851987021833973,
             'v_y': 24.14301225481679, 'v_z': -0.8185070781547807},
            {'x': 27.703486441832403, 'y': 2950.5574264077827, 'z': 17.51916435620088, 'v_x': 13.388186711721229,
             'v_y': 28.92944815545669, 'v_z': -2.18850645855185},
            {'x': 3539.9452480100003, 'y': 1747.4958801684954, 'z': 179.58634245664683, 'v_x': 45.26419373588719,
             'v_y': 48.80994180337515, 'v_z': 2.10194956803325}]
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        # self.att = 2 / ((self.target[0]['x']/1000 - self.uav_init_state[0]['x']/1000) ** 2 +
        #                 (self.target[0]['y']/1000 - self.uav_init_state[0]['y']/1000) ** 2 +
        #                 (self.target[0]['z']/1000 - self.uav_init_state[0]['z']/1000) ** 2)
        # self.rep = 2 / (1 / self.min_sep - 1 / self.min_range) ** 2
        # print(f"att parameter: {self.att}; rep parameter: {self.rep}")
        self.save_path([self.uav_init_state[0]['x'], self.uav_init_state[0]['y'], self.uav_init_state[0]['z']])

    def reset(self):
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        self.path = []
        self.save_path([self.uav_state[0]['x'], self.uav_state[0]['y'], self.uav_state[0]['z']])
        return self.uav_state

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.uav_state, self.target, self.static_obs_state, self.dynamic_obs_state)
        self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step(self, action):
        # pre_tar_dis =  np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
        #                   (self.uav_state[0]['y'] - self.target[0]['y']) ** 2 +
        #                   (self.uav_state[0]['z'] - self.target[0]['z']) ** 2)
        # Calculate the uav's position
        self.uav_state[0]['v_x'] += action[0]['delta_v_x']
        self.uav_state[0]['x'] += self.uav_state[0]['v_x'] * self.delta_t
        self.uav_state[0]['v_y'] += action[0]['delta_v_y']
        self.uav_state[0]['y'] += self.uav_state[0]['v_y'] * self.delta_t
        self.uav_state[0]['v_z'] += action[0]['delta_v_z']
        self.uav_state[0]['z'] += self.uav_state[0]['v_z'] * self.delta_t
        self.save_path([self.uav_state[0]['x'], self.uav_state[0]['y'], self.uav_state[0]['z']])
        # Calculate the dynamic obstacles' position
        for obs in self.dynamic_obs_state:
            obs['x'] += obs['v_x'] * self.delta_t
            obs['y'] += obs['v_y'] * self.delta_t
            obs['z'] += obs['v_z'] * self.delta_t
        # Calculate the reward
        done = False
        reward = 0.0
        static_obs_dis = [np.sqrt(
            (self.static_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.static_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2 +
            (self.static_obs_state[i]['z'] - self.uav_state[0]['z']) ** 2) for i in range(len(self.static_obs_state))]
        dynamic_obs_dis = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2 +
            (self.dynamic_obs_state[i]['z'] - self.uav_state[0]['z']) ** 2) for i in range(len(self.dynamic_obs_state))]
        tar_dis = np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
                          (self.uav_state[0]['y'] - self.target[0]['y']) ** 2 +
                          (self.uav_state[0]['z'] - self.target[0]['z']) ** 2)
        if min(static_obs_dis) > min(dynamic_obs_dis):
            nearest_obs_idx = dynamic_obs_dis.index(min(dynamic_obs_dis))
            self.uav_state[0]['obs_x'] = self.dynamic_obs_state[nearest_obs_idx]['x']
            self.uav_state[0]['obs_y'] = self.dynamic_obs_state[nearest_obs_idx]['y']
            self.uav_state[0]['obs_z'] = self.dynamic_obs_state[nearest_obs_idx]['z']
        else:
            nearest_obs_idx = static_obs_dis.index(min(static_obs_dis))
            self.uav_state[0]['obs_x'] = self.static_obs_state[nearest_obs_idx]['x']
            self.uav_state[0]['obs_y'] = self.static_obs_state[nearest_obs_idx]['y']
            self.uav_state[0]['obs_z'] = self.static_obs_state[nearest_obs_idx]['z']

        urep_static = np.where(np.array(static_obs_dis) < self.min_range,
                               np.exp((-np.array(static_obs_dis))/100), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
                                np.exp((-np.array(dynamic_obs_dis))/100), 0)
        uatt = np.exp(-tar_dis/1000)
        reward += 20 * uatt - sum(urep_static) - sum(urep_dynamic)

        # urep_static = np.where(np.array(static_obs_dis) < self.min_range,
        #                        - 1 / 2 * self.rep * ((1 / np.array(static_obs_dis) - 1 / self.min_range) ** 2), 0)
        # urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
        #                        - 1 / 2 * self.rep * ((1 / np.array(dynamic_obs_dis) - 1 / self.min_range) ** 2), 0)
        # uatt = - self.att * tar_dis
        # reward += sum(urep_static) * 2 + sum(urep_dynamic) * 2 + 100 * uatt

        # reward += -0.1
        # reward += (pre_tar_dis - tar_dis)/10

        if tar_dis < self.min_sep:
            reward += 200.0
            # print(" --------reach the goal")
            done = True
        if np.any(np.array(static_obs_dis) < self.min_sep) or np.any(np.array(dynamic_obs_dis) < self.min_sep):
            reward += -200.0
            # print(" --------collision")
            done = True
        if self.uav_state[0]['x'] < 0.0 or self.uav_state[0]['x'] > 5000.0 or \
            self.uav_state[0]['y'] < 0.0 or self.uav_state[0]['y'] > 5000.0 or \
                self.uav_state[0]['z'] < 0.0 or self.uav_state[0]['z'] > 300.0:
            done = True
        if len(self.path) > self._max_episode_steps:
            done = True

        return self.uav_state, reward, done

    def sample_action(self):
        # # Normal
        # random_delta_v_x = np.random.normal() * 2.0
        # random_delta_v_y = np.random.normal() * 2.0
        # random_delta_v_z = np.random.normal() * 0.3
        # # Mean
        random_delta_v_x = np.random.rand() * 5.0
        random_delta_v_y = np.random.rand() * 5.0
        random_delta_v_z = np.random.rand() * 0.5
        return [{'delta_v_x': random_delta_v_x, 'delta_v_y': random_delta_v_y, 'delta_v_z': random_delta_v_z}]

    def save_path(self, next_state):
        self.path.append(next_state)

    def show_3D_path(self):
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
            # ax.scatter([self.uav_init_state[k]['x'] / 1000], [self.uav_init_state[k]['y'] / 1000],
            #            [self.uav_init_state[k]['z'] / 1000],
            #            color='green', alpha=0.3, marker='x', s=60, label='start')
            # ax.scatter([self.target[k]['x'] / 1000], [self.target[k]['y'] / 1000],
            #            [self.target[k]['z'] / 1000], color='green', alpha=0.7, marker='x', s=60, label='destination')
            ax.plot3D([self.uav_init_state[k]['x'] / 1000, self.target[k]['x'] / 1000],
                      [self.uav_init_state[k]['y'] / 1000, self.target[k]['y'] / 1000],
                      [self.uav_init_state[k]['z'] / 1000, self.target[k]['z'] / 1000],
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

    def show_2D_path(self):

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
            plt.plot([self.uav_init_state[k]['x'] / 1000, self.target[k]['x'] / 1000],
                     [self.uav_init_state[k]['y'] / 1000, self.target[k]['y'] / 1000],
                     color='green', alpha=0.3, linestyle=':')

        plt.title("2D path")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.legend()
        plt.show()

    def plot_reward(self, x, y, z):
        static_obs_dis = [np.sqrt(
            (self.static_obs_state[i]['x'] - x) ** 2 +
            (self.static_obs_state[i]['y'] - y) ** 2 +
            (self.static_obs_state[i]['z'] - z) ** 2) for i in range(len(self.static_obs_state))]
        dynamic_obs_dis = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - x) ** 2 +
            (self.dynamic_obs_state[i]['y'] - y) ** 2 +
            (self.dynamic_obs_state[i]['z'] - z) ** 2) for i in range(len(self.dynamic_obs_state))]
        tar_dis = np.sqrt((x - self.target[0]['x']) ** 2 +
                          (y - self.target[0]['y']) ** 2 +
                          (z - self.target[0]['z']) ** 2)
        urep_static = np.where(np.array(static_obs_dis) < self.min_range,
                               np.exp((-np.array(static_obs_dis))/100), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
                                np.exp((-np.array(dynamic_obs_dis))/100), 0)

        uatt = np.exp(-tar_dis/1000)
        reward = - sum(urep_static) - sum(urep_dynamic) + 20 * uatt

        if tar_dis < self.min_sep:
            reward += 20
        if np.any(np.array(static_obs_dis) < self.min_sep) or np.any(np.array(dynamic_obs_dis) < self.min_sep):
            reward += -20

        return reward


class Viewer(pyglet.window.Window):
    def __init__(self, uav_state, target, static_obs_state, dynamic_obs_state):
        super(Viewer, self).__init__(width=500, height=500, resizable=False, caption='single-agent',
                                     vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.uav_state = uav_state
        self.target = target
        self.uav = pyglet.shapes.Circle(self.uav_state[0]['x'] / 10, self.uav_state[0]['y'] / 10, 3,
                                        color=(1, 100, 1))
        self.target = pyglet.shapes.Circle(self.target[0]['x'] / 10, self.target[0]['y'] / 10, 3, color=(1, 200, 1))
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
        self.uav = pyglet.shapes.Circle(self.uav_state[0]['x'] / 10, self.uav_state[0]['y'] / 10, 3,
                                        color=(1, 100, 1))
        temp = []
        for j in range(len(self.dynamic_obs_state)):
            cur_obs = pyglet.shapes.Circle(self.dynamic_obs_state[j]['x'] / 10, self.dynamic_obs_state[j]['y'] / 10, 3,
                                           color=(249, 109, 249))
            temp.append(cur_obs)
        self.dynamic_obs = temp


if __name__ == "__main__":
    env = SingleContinuousEnv()
    env.reset()

    for i in range(50):
        env.render()
        time.sleep(0.1)
        action = env.sample_action()
        s, r, done = env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]['delta_v_x'], action[0]['delta_v_y'], action[0]['delta_v_z']}\n"
              f"           State: pos {s[0]['x'], s[0]['y'], s[0]['z']};   speed {s[0]['v_x'], s[0]['v_y'], s[0]['v_z']}\n"
              f"           Reward:{r}\n")
        if done:
            env.show_path()
            break

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
