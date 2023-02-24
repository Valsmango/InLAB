# coding=utf-8
import time
import pyglet
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import copy

'''
state设定：【x，y，z，v_x，v_y，v_z，tar_x，tar_y，tar_z，obs_x，obs_y，obs_z】

[{'x': 0, 'y': 0, 'z': 0, 'v_x': 100, 'v_y': 100, 'v_z': 6, 'tar_x': 5000, 'tar_y': 5000, 'tar_z': 300, 'obs_x': 0, 'obs_y': 0, 'obs_z': 0}]
[{'x': 0, 'y': 0, 'z': 0, 'v_hori': 100 * np.sqrt(2), 'v_vert': 6, 'angle_hori': (1 / 4) * np.pi}]

action设定：【dvx， dvy， dvz】
'''


class SingleContinuousEnv(object):
    def __init__(self):
        self.env_kinds = 4
        self.env_id = 0
        self._max_episode_steps = 200
        self.model = None
        self.randomly_choose_env()

    def randomly_choose_env(self):
        # randomly choose an env
        # self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                   init_start=[{'x': 0, 'y': 0, 'z': 150,
        #                                                'v_x': 100, 'v_y': 100, 'v_z': 0,
        #                                                'tar_x': 5000, 'tar_y': 5000, 'tar_z': 150,
        #                                                'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                   init_target=[{'x': 5000, 'y': 5000, 'z': 150}])
        # self.env_id = -1

        # 随机起点和终点
        x = np.random.rand() * 300  # 避免一步就无了
        y = np.random.rand() * 5000
        z = np.random.rand() * 200 + 50
        tar_x = np.random.rand() * 300 + 4700
        tar_y = np.random.rand() * 5000
        tar_z = np.random.rand() * 200 + 50
        self.model = SingleContinuousEnv0(self._max_episode_steps,
                                          init_start=[{'x': x, 'y': y, 'z': z,
                                                       'v_x': (tar_x - x) / 50, 'v_y': (tar_y - y) / 50, 'v_z': (tar_z - z) / 50,
                                                       'tar_x': tar_x, 'tar_y': tar_y, 'tar_z': tar_z,
                                                       'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
                                          init_target=[{'x': tar_x, 'y': tar_y, 'z': tar_z}])

        # 四种环境
        # tmp = np.random.rand()
        # if tmp < 0.25:
        #     self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                       init_start=[{'x': 0, 'y': 0, 'z': 150,
        #                                                    'v_x': 100, 'v_y': 100, 'v_z': 0,
        #                                                    'tar_x': 5000, 'tar_y': 5000, 'tar_z': 150,
        #                                                    'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                       init_target=[{'x': 5000, 'y': 5000, 'z': 150}])
        #     self.env_id = 1
        # elif tmp < 0.5:
        #     self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                       init_start=[{'x': 5000.0, 'y': 0.0, 'z': 150,
        #                                                    'v_x': -100.0, 'v_y': 100.0, 'v_z': 0,
        #                                                    'tar_x': 0.0, 'tar_y': 5000.0, 'tar_z': 200,
        #                                                    'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                       init_target=[{'x': 0.0, 'y': 5000.0, 'z': 200}])
        #     self.env_id = 2
        # elif tmp < 0.75:
        #     self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                       init_start=[{'x': 0.0, 'y': 5000.0, 'z': 150,
        #                                                    'v_x': 100.0, 'v_y': -100.0, 'v_z': 0,
        #                                                    'tar_x': 5000.0, 'tar_y': 0.0, 'tar_z': 100,
        #                                                    'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                       init_target=[{'x': 5000.0, 'y': 0.0, 'z': 100}])
        #     self.env_id = 3
        # else:
        #     self.model = SingleContinuousEnv0(self._max_episode_steps,
        #                                       init_start=[{'x': 0.0, 'y': 2500.0, 'z': 150,
        #                                                    'v_x': 100.0, 'v_y': 0.0, 'v_z': 0.0,
        #                                                    'tar_x': 5000.0, 'tar_y': 2500.0, 'tar_z': 150.0,
        #                                                    'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
        #                                       init_target=[{'x': 5000.0, 'y': 2500.0, 'z': 150.0}])
        #     self.env_id = 4

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
        self.model.show_xy_path()
        self.model.show_xz_path()

    def close(self):
        self.model.close()

    def plot_reward(self, x, y, z=150):
        return self.model.plot_reward(x, y, z)

    def get_env_id(self):
        return self.env_id


class SingleContinuousEnv0(object):
    def __init__(self, max_episode_steps, init_start, init_target):
        self.uav_state = []
        self.static_obstacle_num = 20
        self.dynamic_obstacle_num = 2
        self.max_action = np.array([5.0, 5.0, 0.5])
        # ndarray - tuple
        self.static_obs_state = []
        self.dynamic_obs_state = []
        self.uav_init_state = self.uav_state
        self.dynamic_obs_init_state = self.dynamic_obs_state
        self.path = []
        self.delta_t = 1
        self.min_sep_hori = 152.4
        self.min_sep_vert = 30.48
        self.min_range = 1000
        self.viewer = None
        self._max_episode_steps = max_episode_steps

        self.init_map(init_start, init_target)

    def init_map(self, init_start, init_target):
        self.target = init_target
        self.uav_init_state = init_start
        self.static_obs_state = [{'x': 3230.388590643148, 'y': 2933.2656458711062, 'z': 113.4210720886783, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4207.45927343738, 'y': 1744.5154956120791, 'z': 32.391479943413316, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2964.4695907062214, 'y': 2858.225678987416, 'z': 106.2198924612835, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1471.0258353243426, 'y': 796.519928285075, 'z': 26.46502981349346, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3619.8530503544594, 'y': 722.4057727638447, 'z': 263.12337789494603, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2773.7969443424145, 'y': 3644.821538096674, 'z': 293.20860838957805, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1904.000584931992, 'y': 3649.24674503862, 'z': 88.41635571633817, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 653.5854038558919, 'y': 2800.980047837858, 'z': 147.6326500378459, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3620.9688394896198, 'y': 3554.331829624033, 'z': 90.73407967812005, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1211.222102657049, 'y': 1110.2417872537092, 'z': 171.49927078391153, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3111.849570751636, 'y': 4284.930532062208, 'z': 261.417609114504, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1142.8316746659793, 'y': 2561.1549908696597, 'z': 192.95615988835928, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 919.7628517663965, 'y': 1264.1817763605113, 'z': 119.21451593236858, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2308.5084459506143, 'y': 4044.8845531222078, 'z': 116.04478285793579, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 625.4502261660186, 'y': 2696.146587717727, 'z': 154.00082449798398, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2190.0192218172706, 'y': 3929.8497110290696, 'z': 155.65679641102804, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 879.5043323029183, 'y': 2673.6300254998314, 'z': 113.97700065962208, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1071.6764400543734, 'y': 1113.9208846073632, 'z': 36.228686035711, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 848.6311676517016, 'y': 1275.9706067754823, 'z': 246.7809938722449, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1769.0495561848122, 'y': 2393.652032532954, 'z': 97.82548391633325, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1845.3210981166665, 'y': 1506.4017160984736, 'z': 87.60506807038944, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3508.777490661421, 'y': 3962.509247313749, 'z': 169.4013496649088, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 750.3238968877599, 'y': 1685.0164004877074, 'z': 236.08929573111354, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3763.533564632695, 'y': 2038.495788451939, 'z': 132.40449330425682, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 944.6353461123591, 'y': 4437.583959355806, 'z': 266.7422180633163, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 634.4140487020544, 'y': 1742.0462471234316, 'z': 87.14270686268593, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3196.0729480662276, 'y': 3633.2429084784344, 'z': 188.91542118950298, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1851.5130072583931, 'y': 921.0104734175779, 'z': 137.17332107847042, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4096.181382245324, 'y': 2075.9815134036035, 'z': 19.05877017551828, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3155.937643964505, 'y': 2049.493982398455, 'z': 180.51184359046633, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2398.537961948024, 'y': 3961.2937561516515, 'z': 174.0688391923668, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3481.997238418981, 'y': 3342.8087407561256, 'z': 198.11733436109319, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 969.9555641627869, 'y': 1029.198678754819, 'z': 106.0473882876473, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4438.759401965695, 'y': 928.2506459539111, 'z': 186.59584523129868, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4464.608208328604, 'y': 3090.549126101325, 'z': 249.90584210971045, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1607.3378475344189, 'y': 3746.015055700696, 'z': 49.298078411637306, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3202.5617928631254, 'y': 1874.6241189732634, 'z': 200.0707831998786, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 739.5870053876283, 'y': 3797.6269962045317, 'z': 48.57350110653683, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 838.279909692599, 'y': 2698.6249626807275, 'z': 72.13945946877803, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3005.4292118772264, 'y': 1841.9583999485903, 'z': 60.41711103427802, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2329.8522355341593, 'y': 927.7819795575692, 'z': 280.4483487314923, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1649.9820742508416, 'y': 1195.6697946481931, 'z': 186.70658675122064, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 629.7237449983477, 'y': 1637.0090857172265, 'z': 219.0855957078834, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2777.2410652025255, 'y': 1576.2292546317412, 'z': 43.590975220825456, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1627.5124081624756, 'y': 3632.6199049407837, 'z': 143.68541277263537, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1691.4372781952625, 'y': 3211.3146353899588, 'z': 208.80103993027348, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3528.2085805528804, 'y': 2726.5245408448345, 'z': 197.5801734120882, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1603.7232420787016, 'y': 4477.020840503068, 'z': 146.30141887504777, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1396.9050005059803, 'y': 3137.356305527304, 'z': 221.7681332190756, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2433.090343504532, 'y': 4431.372997852416, 'z': 42.87351747529318, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4408.62896130754, 'y': 1823.7688221442272, 'z': 133.39350718631493, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1693.0679499142952, 'y': 4343.0944459997845, 'z': 125.32086945935423, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2379.1500027223183, 'y': 3923.6278765339416, 'z': 37.62759847317702, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3735.014168410979, 'y': 3399.7837933392043, 'z': 258.25126073654025, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3900.778100577414, 'y': 4445.037777982356, 'z': 287.71331150945974, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2145.5438808320064, 'y': 3726.9408865640025, 'z': 131.8126429953404, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2473.5239115732456, 'y': 2266.9871969734186, 'z': 155.60981210212105, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1587.35642882512, 'y': 3454.6387452838508, 'z': 24.368706561145625, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3840.1357088183627, 'y': 2251.380425112339, 'z': 107.34316697687987, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2220.228016609292, 'y': 880.5569731842197, 'z': 96.29536235723776, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2192.1926932229558, 'y': 4025.3648221445724, 'z': 139.30779912878168, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3490.7827851419424, 'y': 3710.42116301576, 'z': 164.7500188102599, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 651.8855611058423, 'y': 3463.6715078730717, 'z': 164.0579509332575, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3653.1127303979065, 'y': 3650.126855492514, 'z': 173.06465045307445, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1061.0757843958304, 'y': 1439.353545883001, 'z': 128.14584543721057, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 720.9578077809624, 'y': 3976.4526847213706, 'z': 117.4100409442279, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1142.6263069786132, 'y': 1445.1689869710026, 'z': 248.10398177859275, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1983.9744883260616, 'y': 3403.038951521122, 'z': 195.78692969183788, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3798.3597144297014, 'y': 2650.905887698258, 'z': 210.13072086081868, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2563.0098750836983, 'y': 4101.8925940154295, 'z': 283.644957102515, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3719.3406249306254, 'y': 904.2793009777643, 'z': 295.6380534467112, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2649.4761730941395, 'y': 4068.2089080682144, 'z': 19.35202913141746, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2703.2496450894128, 'y': 4179.680261279273, 'z': 153.9732279847234, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4466.934868216597, 'y': 847.7659946478964, 'z': 89.67201802700389, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2729.053889272036, 'y': 839.2460142271015, 'z': 23.023401379016008, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1441.4233970845062, 'y': 1711.9713481283623, 'z': 243.3125669233107, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3415.2569155340425, 'y': 2803.4792190090516, 'z': 230.7717576510526, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3908.1447544314015, 'y': 515.6155882251769, 'z': 253.99903583230403, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3495.8939575887234, 'y': 2325.658598979039, 'z': 32.152099320149006, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4437.443202106157, 'y': 2070.2050835425616, 'z': 60.302310594854305, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}]
        self.dynamic_obs_init_state = [{'x': 2768.5869315567375, 'y': 2530.199651453616, 'z': 196.1140528735404, 'v_x': 16.254193622126607, 'v_y': 10.756218999711159, 'v_z': -1.0437859596440386}, {'x': 3658.9764465324824, 'y': 3088.414553421304, 'z': 179.55571499987548, 'v_x': 10.186299601729734, 'v_y': 2.3364461540645474, 'v_z': -0.11478783355873112}, {'x': 1228.4001260189643, 'y': 1723.758345973561, 'z': 160.53697889133534, 'v_x': 7.63099545748206, 'v_y': 16.823256425441414, 'v_z': -1.2655404463268627}, {'x': 2524.0920525658553, 'y': 3454.0775146116825, 'z': 92.90479637360954, 'v_x': 18.18786411461985, 'v_y': 10.43880041528583, 'v_z': -1.141711877126959}, {'x': 1127.6595766254, 'y': 3305.5272029369894, 'z': 208.7456932831874, 'v_x': 11.014644481752114, 'v_y': 10.616396361521799, 'v_z': -0.3244448032630425}, {'x': 2395.906380205874, 'y': 506.38171744423755, 'z': 40.523982152937336, 'v_x': 19.2033395919542, 'v_y': 22.659157782878328, 'v_z': 0.9574744608311176}, {'x': 2583.426485795169, 'y': 706.4717733134298, 'z': 14.077732825168443, 'v_x': 7.946365133352754, 'v_y': 11.901751033056504, 'v_z': -1.0329027419192711}, {'x': 3790.335930633661, 'y': 4124.801638531565, 'z': 57.2621069538656, 'v_x': 6.901714898447469, 'v_y': 19.371623625778827, 'v_z': -0.9444177579914449}, {'x': 1448.1903497797562, 'y': 986.6681817547902, 'z': 153.8475351329597, 'v_x': 5.677378079308474, 'v_y': 4.2581382071785745, 'v_z': -0.47088653663189906}, {'x': 1438.458985314864, 'y': 2066.1091486859887, 'z': 169.9685846841342, 'v_x': 10.230640732317637, 'v_y': 2.334142923989449, 'v_z': -0.6374269803622354}, {'x': 2637.053045323583, 'y': 4484.472070538564, 'z': 23.037953323707683, 'v_x': 4.774994939583332, 'v_y': 22.485361988330176, 'v_z': -0.48262111902765437}, {'x': 515.0362961092671, 'y': 690.6126471824181, 'z': 254.00275151526893, 'v_x': 15.143716682486605, 'v_y': 6.933397422767521, 'v_z': -0.703605462924817}, {'x': 3900.2227892754217, 'y': 2245.8204844518264, 'z': 123.28916888133905, 'v_x': 1.2363737259793646, 'v_y': 11.410022653220924, 'v_z': 1.4844731282442676}, {'x': 3912.4803561652516, 'y': 4463.8951516344205, 'z': 169.1206400488795, 'v_x': 5.32775601586091, 'v_y': 12.521414117519436, 'v_z': 0.0694317673121514}, {'x': 4007.084530529808, 'y': 4165.450373043117, 'z': 261.32799053622244, 'v_x': 1.984641474136578, 'v_y': 23.045005289051396, 'v_z': -0.16330886420615087}, {'x': 2682.710435606956, 'y': 1634.475346266522, 'z': 268.7093210021009, 'v_x': 2.435939765518702, 'v_y': 9.109843875455482, 'v_z': 0.32057019490978034}, {'x': 831.2230701454157, 'y': 4223.635503754798, 'z': 139.35669058025263, 'v_x': 18.618686990538688, 'v_y': 2.8074099724593014, 'v_z': -1.4736823161702177}, {'x': 3960.432203052959, 'y': 4103.708664532982, 'z': 95.28515204108379, 'v_x': 12.498389240633717, 'v_y': 13.179520344236604, 'v_z': -1.0200315420753308}, {'x': 1487.4314221403133, 'y': 4025.9521845071627, 'z': 261.7282655575755, 'v_x': 5.773109088237799, 'v_y': 20.74871856634153, 'v_z': -0.7315889211551558}, {'x': 4077.5214115412296, 'y': 730.9609290067631, 'z': 61.253195251766456, 'v_x': 17.424553527224496, 'v_y': 2.625806063254238, 'v_z': -0.19044400194808908}]
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        self.att = 2 / ((self.target[0]['x']/1000 - self.uav_init_state[0]['x']/1000) ** 2 +
                        (self.target[0]['y']/1000 - self.uav_init_state[0]['y']/1000) ** 2 +
                        (self.target[0]['z']/1000 - self.uav_init_state[0]['z']/1000) ** 2)
        self.rep = 2 / (1 / self.min_sep_hori - 1 / self.min_range) ** 2
        # print(f"att parameter: {self.att}; rep parameter: {self.rep}")
        self.save_path([self.uav_init_state[0]['x'], self.uav_init_state[0]['y'], self.uav_init_state[0]['z']])

        # update the nearest obs info
        static_obs_dis = [np.sqrt(
            (self.static_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.static_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2 +
            (self.static_obs_state[i]['z'] - self.uav_state[0]['z']) ** 2) for i in range(len(self.static_obs_state))]
        dynamic_obs_dis = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2 +
            (self.dynamic_obs_state[i]['z'] - self.uav_state[0]['z']) ** 2) for i in range(len(self.dynamic_obs_state))]
        if min(static_obs_dis) > min(dynamic_obs_dis):
            nearest_obs_idx = dynamic_obs_dis.index(min(dynamic_obs_dis))
            self.uav_state[0]['obs_x'] = self.dynamic_obs_state[nearest_obs_idx]['x']
            self.uav_state[0]['obs_y'] = self.dynamic_obs_state[nearest_obs_idx]['y']
            self.uav_state[0]['obs_z'] = self.dynamic_obs_state[nearest_obs_idx]['z']
            self.uav_init_state[0]['obs_x'] = self.uav_state[0]['obs_x']
            self.uav_init_state[0]['obs_y'] = self.uav_state[0]['obs_y']
            self.uav_init_state[0]['obs_z'] = self.uav_state[0]['obs_z']
        else:
            nearest_obs_idx = static_obs_dis.index(min(static_obs_dis))
            self.uav_state[0]['obs_x'] = self.static_obs_state[nearest_obs_idx]['x']
            self.uav_state[0]['obs_y'] = self.static_obs_state[nearest_obs_idx]['y']
            self.uav_state[0]['obs_z'] = self.static_obs_state[nearest_obs_idx]['z']
            self.uav_init_state[0]['obs_x'] = self.uav_state[0]['obs_x']
            self.uav_init_state[0]['obs_y'] = self.uav_state[0]['obs_y']
            self.uav_init_state[0]['obs_z'] = self.uav_state[0]['obs_z']


    def reset(self):
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        self.path = []
        self.save_path([self.uav_state[0]['x'], self.uav_state[0]['y'], self.uav_state[0]['z']])
        # 【x，y，z，v_x，v_y，v_z，tar_x，tar_y，tar_z，obs_x，obs_y，obs_z】
        return_state = copy.deepcopy(self.uav_state)
        return_state[0]['x'] = return_state[0]['x'] / 5000.0
        return_state[0]['y'] = return_state[0]['y'] / 5000.0
        return_state[0]['z'] = return_state[0]['z'] / 300.0
        return_state[0]['v_x'] = return_state[0]['v_x'] / 200.0
        return_state[0]['v_y'] = return_state[0]['v_y'] / 200.0
        return_state[0]['v_z'] = return_state[0]['v_z'] / 10.0
        return_state[0]['tar_x'] = return_state[0]['tar_x'] / 5000.0
        return_state[0]['tar_y'] = return_state[0]['tar_y'] / 5000.0
        return_state[0]['tar_z'] = return_state[0]['tar_z'] / 300.0
        return_state[0]['obs_x'] = return_state[0]['obs_x'] / 5000.0
        return_state[0]['obs_y'] = return_state[0]['obs_y'] / 5000.0
        return_state[0]['obs_z'] = return_state[0]['obs_z'] / 300.0
        return return_state

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.uav_state, self.target, self.static_obs_state, self.dynamic_obs_state)
        self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step(self, input_action):
        action = copy.deepcopy(input_action)
        action[0]['delta_v_x'] = action[0]['delta_v_x'] * self.max_action[0]
        action[0]['delta_v_y'] = action[0]['delta_v_y'] * self.max_action[1]
        action[0]['delta_v_z'] = action[0]['delta_v_z'] * self.max_action[2]
        done = False
        reward = 0.0

        # pre_tar_dis =  np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
        #                   (self.uav_state[0]['y'] - self.target[0]['y']) ** 2 +
        #                   (self.uav_state[0]['z'] - self.target[0]['z']) ** 2)
        # Calculate the uav's position
        self.uav_state[0]['v_x'] += action[0]['delta_v_x']
        if self.uav_state[0]['v_x'] > 200:
            self.uav_state[0]['v_x'] = 200
            reward += -1
        elif self.uav_state[0]['v_x'] < -200:
            self.uav_state[0]['v_x'] = -200
            reward += -1
        self.uav_state[0]['x'] += self.uav_state[0]['v_x'] * self.delta_t
        self.uav_state[0]['v_y'] += action[0]['delta_v_y']
        if self.uav_state[0]['v_y'] > 200:
            self.uav_state[0]['v_y'] = 200
            reward += -1
        elif self.uav_state[0]['v_y'] < -200:
            self.uav_state[0]['v_y'] = -200
            reward += -1
        self.uav_state[0]['y'] += self.uav_state[0]['v_y'] * self.delta_t
        self.uav_state[0]['v_z'] += action[0]['delta_v_z']
        if self.uav_state[0]['v_z'] > 10:
            self.uav_state[0]['v_z'] = 10
            reward += -1
        elif self.uav_state[0]['v_z'] < -10:
            self.uav_state[0]['v_z'] = -10
            reward += -1
        self.uav_state[0]['z'] += self.uav_state[0]['v_z'] * self.delta_t
        self.save_path([self.uav_state[0]['x'], self.uav_state[0]['y'], self.uav_state[0]['z']])
        # Calculate the dynamic obstacles' position
        for obs in self.dynamic_obs_state:
            obs['x'] += obs['v_x'] * self.delta_t
            obs['y'] += obs['v_y'] * self.delta_t
            obs['z'] += obs['v_z'] * self.delta_t
        # Calculate the reward
        static_obs_dis = [np.sqrt(
            (self.static_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.static_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2 +
            (self.static_obs_state[i]['z'] - self.uav_state[0]['z']) ** 2) for i in range(len(self.static_obs_state))]
        static_obs_dis_hori = [np.sqrt(
            (self.static_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.static_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2) for i in range(len(self.static_obs_state))]
        static_obs_dis_vert = [np.abs(self.static_obs_state[i]['z'] - self.uav_state[0]['z']) for i in range(len(self.static_obs_state))]
        dynamic_obs_dis = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2 +
            (self.dynamic_obs_state[i]['z'] - self.uav_state[0]['z']) ** 2) for i in range(len(self.dynamic_obs_state))]
        dynamic_obs_dis_hori = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2) for i in range(len(self.dynamic_obs_state))]
        dynamic_obs_dis_vert = [np.abs(self.dynamic_obs_state[i]['z'] - self.uav_state[0]['z']) for i in range(len(self.dynamic_obs_state))]
        tar_dis = np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
                          (self.uav_state[0]['y'] - self.target[0]['y']) ** 2 +
                          (self.uav_state[0]['z'] - self.target[0]['z']) ** 2)
        tar_dis_hori = np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
                          (self.uav_state[0]['y'] - self.target[0]['y']) ** 2)
        tar_dis_vert = np.abs(self.uav_state[0]['z'] - self.target[0]['z'])
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

        # urep_static = np.where(np.array(static_obs_dis) < self.min_range,
        #                        np.exp((-np.array(static_obs_dis))/100), 0)
        # urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
        #                         np.exp((-np.array(dynamic_obs_dis))/100), 0)
        # uatt = np.exp(-tar_dis/5000)
        # reward += uatt
        # reward += 20 * uatt - sum(urep_static) - sum(urep_dynamic)

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
        # if tar_dis < 152.4:
            reward += 100
            # print(" --------reach the goal")
            done = True
        elif np.any(np.array(np.array(static_obs_dis_vert) < self.min_sep_vert) &
                    np.array(np.array(static_obs_dis_hori) < self.min_sep_hori)) or \
                np.any(np.array(np.array(dynamic_obs_dis_vert) < self.min_sep_vert) &
                    np.array(np.array(dynamic_obs_dis_hori) < self.min_sep_hori)):
            reward += -100
            # print(" --------collision")
            done = True
        elif self.uav_state[0]['x'] < 0.0 or self.uav_state[0]['x'] > 5000.0 or \
            self.uav_state[0]['y'] < 0.0 or self.uav_state[0]['y'] > 5000.0 or \
                self.uav_state[0]['z'] < 0.0 or self.uav_state[0]['z'] > 300.0:
            reward += -100
            # print(" --------out of the boundary")
            done = True
        elif len(self.path) > self._max_episode_steps:
            done = True

        return_state = copy.deepcopy(self.uav_state)
        return_state[0]['x'] = return_state[0]['x'] / 5000.0
        return_state[0]['y'] = return_state[0]['y'] / 5000.0
        return_state[0]['z'] = return_state[0]['z'] / 300.0
        return_state[0]['v_x'] = return_state[0]['v_x'] / 200.0
        return_state[0]['v_y'] = return_state[0]['v_y'] / 200.0
        return_state[0]['v_z'] = return_state[0]['v_z'] / 10.0
        return_state[0]['tar_x'] = return_state[0]['tar_x'] / 5000.0
        return_state[0]['tar_y'] = return_state[0]['tar_y'] / 5000.0
        return_state[0]['tar_z'] = return_state[0]['tar_z'] / 300.0
        return_state[0]['obs_x'] = return_state[0]['obs_x'] / 5000.0
        return_state[0]['obs_y'] = return_state[0]['obs_y'] / 5000.0
        return_state[0]['obs_z'] = return_state[0]['obs_z'] / 300.0
        return return_state, reward, done

    def sample_action(self):
        # # Mean
        random_delta_v_x = np.random.rand()
        random_delta_v_y = np.random.rand()
        random_delta_v_z = np.random.rand()
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

    def show_xy_path(self):
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

        circle = plt.Circle(xy=([self.path[path_len - 1][0] / 1000], [self.path[path_len - 1][1] / 1000]),
                            radius=self.min_sep_hori / 1000, linestyle=':', color='green', fill=False)
        # ax = plt.gcf().gca()
        # ax.add_artist(circle)
        plt.gca().add_patch(circle)

        # plt.axis('equal')
        plt.title("2D path - xy")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.legend()
        plt.show()

    def show_xz_path(self):
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
            plt.plot([self.uav_init_state[k]['x'] / 1000, self.target[k]['x'] / 1000],
                     [self.uav_init_state[k]['z'] / 1000, self.target[k]['z'] / 1000],
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

    def plot_reward(self, x, y, z):
        reward = 0.0
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
        # urep_static = np.where(np.array(static_obs_dis) < self.min_range,
        #                        np.exp((-np.array(static_obs_dis))/100), 0)
        # urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
        #                         np.exp((-np.array(dynamic_obs_dis))/100), 0)
        # uatt = np.exp(-tar_dis/5000)
        # reward += - sum(urep_static) - sum(urep_dynamic)
        # reward += uatt

        urep_static = np.where(np.array(static_obs_dis) < self.min_range,
                               - 1 / 2 * self.rep * (1 / np.array(static_obs_dis) - 1 / self.min_range), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
                               - 1 / 2 * self.rep * (1 / np.array(dynamic_obs_dis) - 1 / self.min_range), 0)
        uatt = - self.att * tar_dis
        # reward += 0.00125*sum(urep_static) + 0.00125*sum(urep_dynamic)
        reward += 0.0025*uatt

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
    init_state = env.reset()

    # for i in range(50):
    #     env.render()
    #     time.sleep(0.1)
    #     action = env.sample_action()
    #     s, r, done = env.step(action)
    #     print(f"currently, the {i + 1} step:\n"
    #           f"           Action: speed {action[0]['delta_v_x']*5.0, action[0]['delta_v_y']*5.0, action[0]['delta_v_z']*0.5}\n"
    #           f"           State: pos {s[0]['x']*5000.0, s[0]['y']*5000.0, s[0]['z']*300.0};   speed {s[0]['v_x']*200, s[0]['v_y']*200.0, s[0]['v_z']*10}\n"
    #           f"           Reward:{r}\n")
    #
    #     if done:
    #         env.show_path()
    #         break

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

    x = np.arange(-500, 5500, 10)
    y = np.arange(-500, 5500, 10)
    X, Y = np.meshgrid(x, y)
    Z = env.plot_reward(X, Y)
    ct = plt.contour(X, Y, Z, 100)
    plt.clabel(ct, inline=True)
    plt.show()
