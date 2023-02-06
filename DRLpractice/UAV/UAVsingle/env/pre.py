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


class SingleContinuousEnv_pre(object):
    def __init__(self):
        self.env_kinds = 4
        self.env_id = 0
        self._max_episode_steps = 100
        self.model = None
        self.randomly_choose_env()

    def randomly_choose_env(self):
        # randomly choose an env
        self.model = SingleContinuousEnv0(self._max_episode_steps,
                                          init_start=[{'x': 0, 'y': 0, 'z': 150,
                                                       'v_x': 100, 'v_y': 100, 'v_z': 0,
                                                       'tar_x': 5000, 'tar_y': 5000, 'tar_z': 150,
                                                       'obs_x': 0, 'obs_y': 0, 'obs_z': 0}],
                                          init_target=[{'x': 5000, 'y': 5000, 'z': 150}])
        self.env_id = 0

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

    def plot_reward(self, x, y, z=0):
        return self.model.plot_reward(x, y, z)

    def get_env_id(self):
        return self.env_id


class SingleContinuousEnv0_pre(object):
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
        self.min_sep_hori = 100.0
        self.min_sep_vert = 10.0
        self.min_range = 1000
        self.viewer = None
        self._max_episode_steps = max_episode_steps

        self.init_map(init_start, init_target)

    def init_map(self, init_start, init_target):
        self.target = init_target
        self.uav_init_state = init_start
        self.static_obs_state = [{'x': 2547.307269139552, 'y': 1545.0670498980771, 'z': 295.3136127157938, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3939.0494323671737, 'y': 3789.6411194061247, 'z': 243.78728253592206, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3941.3716076721544, 'y': 4286.2750795977345, 'z': 43.2247797505332, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2519.9304979261406, 'y': 2487.586197743359, 'z': 173.48882590123463, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4140.195584191861, 'y': 2842.5683095241197, 'z': 157.98847671705894, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3076.553265747653, 'y': 4281.377014390158, 'z': 14.609444415524598, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2559.517853241137, 'y': 3538.232380265296, 'z': 249.76023443840336, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4068.224768264749, 'y': 3901.616667466977, 'z': 248.92227850735185, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 747.9317040436828, 'y': 3087.1352477135943, 'z': 21.960928045669593, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2604.888725081889, 'y': 588.7899695404002, 'z': 78.97016274555959, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2544.597104887479, 'y': 2579.5574512717753, 'z': 281.81053373015726, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4196.613519166605, 'y': 855.7257021721209, 'z': 69.94046286250475, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2911.572362876407, 'y': 2013.4775410041361, 'z': 92.82650575240552, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3667.251214156835, 'y': 1889.52173823538, 'z': 211.026734113371, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3788.8067563314294, 'y': 2097.320247373168, 'z': 92.7816702558911, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3658.765259142098, 'y': 2394.3561265140056, 'z': 144.237369705236, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 939.6264280071, 'y': 2572.5531421156325, 'z': 182.721934631846, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2852.792314990716, 'y': 2480.663789122596, 'z': 47.274462211544886, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2297.558169129761, 'y': 1920.6035808287152, 'z': 122.99156004549447, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4447.323440109625, 'y': 1594.0340740178979, 'z': 28.72222287305679, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3564.3716943022528, 'y': 4172.249307887742, 'z': 254.3309873400763, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3774.427399971537, 'y': 580.121968568295, 'z': 191.47106725376148, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1993.6305490875993, 'y': 1847.2586366233936, 'z': 122.17495419550167, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1017.3177400073441, 'y': 818.3871754917909, 'z': 235.1486382927932, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3984.532024379963, 'y': 2762.675289643419, 'z': 279.4534222295948, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2503.7250270813756, 'y': 1747.108476471327, 'z': 134.31830126072794, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3775.6954594333524, 'y': 4064.2063054021946, 'z': 204.13382569194843, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1522.5307431829726, 'y': 2565.245136261253, 'z': 271.24496980595603, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2005.2055776769193, 'y': 1220.2751209890152, 'z': 284.4852136000808, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1508.5534065253642, 'y': 2388.986420540822, 'z': 196.80717274301583, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1225.5974905193136, 'y': 3205.3842610625356, 'z': 57.19653199967222, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4074.255719353994, 'y': 788.3072127051798, 'z': 266.2491737939294, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2470.543195625156, 'y': 3607.521715488451, 'z': 143.44878193345244, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3292.2066021468777, 'y': 3353.7745117837985, 'z': 3.8219476703554744, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2335.7409609150777, 'y': 4272.651804817704, 'z': 233.75457546851038, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3441.109431911402, 'y': 963.4446075924136, 'z': 284.4029549100709, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1158.1121752528566, 'y': 1056.2366058746375, 'z': 137.16407887551344, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2854.5904165567877, 'y': 2088.034647268514, 'z': 81.62865104528623, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3759.7304126477334, 'y': 4265.411872756378, 'z': 2.3048786867373106, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3543.276682519022, 'y': 2180.991534138425, 'z': 66.94467901432854, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3437.416801702158, 'y': 3629.4724295571696, 'z': 271.71584341088897, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 541.7341633428947, 'y': 872.6590794905933, 'z': 51.03690066823498, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3336.5130864974594, 'y': 4237.377966064094, 'z': 108.30133916854139, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2679.7169647845753, 'y': 590.9315589619774, 'z': 278.96676914103995, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2180.4681768699347, 'y': 2221.2159528543457, 'z': 257.36706479592567, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1750.1647179503018, 'y': 1977.9634515966045, 'z': 37.293798349307636, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3575.807220723719, 'y': 2501.306780735752, 'z': 274.6226404879244, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3723.575100700567, 'y': 565.0319884367656, 'z': 271.02877369757095, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2604.5176623434554, 'y': 2372.8440969740523, 'z': 190.66414250815146, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1925.7137877965388, 'y': 931.7199095571938, 'z': 147.56485378086893, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3331.333756416439, 'y': 1855.3741970327305, 'z': 87.47896872798773, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3410.946453905023, 'y': 4477.078881832562, 'z': 256.8504602534793, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2972.5327455334145, 'y': 2082.794594692815, 'z': 296.68033594613325, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 510.1685670027978, 'y': 3637.1415079817684, 'z': 31.586282377764498, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1581.685929080736, 'y': 942.0916073239596, 'z': 26.48473472148992, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2276.4910226889965, 'y': 3553.8881478363755, 'z': 118.86304072822755, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3491.7898015017845, 'y': 2235.9040605423356, 'z': 290.04419811345986, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1976.914521034002, 'y': 909.3085376289794, 'z': 264.0677523238379, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2548.261262727068, 'y': 4288.196873834577, 'z': 40.24098838887677, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 906.1049720884084, 'y': 548.116601035674, 'z': 197.3415823828747, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3299.6977928356655, 'y': 3210.6137551659826, 'z': 148.26107619681113, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2103.759381752275, 'y': 3863.4587974340875, 'z': 67.89419580100855, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1477.9662581626023, 'y': 933.8389085473948, 'z': 117.08130090344923, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1642.1996117152835, 'y': 3440.832785466654, 'z': 112.0042303287479, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 672.6855351344403, 'y': 2352.854558861787, 'z': 290.01598234749554, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2899.545742059243, 'y': 1845.9589852171282, 'z': 39.88091618577012, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1492.249617512778, 'y': 2518.2912416345425, 'z': 115.54154703781958, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1438.3848607523337, 'y': 1935.2372080776554, 'z': 99.46240707627214, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1374.0344087908722, 'y': 2626.67941641888, 'z': 41.79033906027515, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1519.8545179475568, 'y': 3361.3370955273904, 'z': 267.58037001669175, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1313.1445919608598, 'y': 3299.3919501194264, 'z': 78.92278876161204, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2210.7102680067283, 'y': 1781.7731823778922, 'z': 255.15743986241608, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3001.782601206183, 'y': 892.7402383174274, 'z': 10.484485819098788, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3891.5704485042543, 'y': 4352.158844946154, 'z': 253.85895488577987, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2868.9345290636056, 'y': 3718.5340643111167, 'z': 104.62146331514653, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2626.1442762770257, 'y': 1499.774580349082, 'z': 291.67438159978883, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1643.2115546891444, 'y': 2106.0546501408894, 'z': 210.10436398897428, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2932.490562809095, 'y': 1098.9572375517455, 'z': 227.55387331795103, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 744.0575128153282, 'y': 1139.3304319674967, 'z': 257.60272535623915, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 853.009761727765, 'y': 1608.3522023097553, 'z': 41.23382332396665, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1877.9028913319364, 'y': 1486.17163063108, 'z': 15.81817878910683, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2825.098909474757, 'y': 2124.625410945534, 'z': 163.7055854909377, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1475.1913743404043, 'y': 4280.1502113588995, 'z': 76.0320679777426, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1204.044135817106, 'y': 2753.460645875113, 'z': 162.65991686870817, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 2399.0013029175498, 'y': 841.0276885311534, 'z': 156.55612904971503, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3653.348231815458, 'y': 2876.8019078631964, 'z': 145.87587568648507, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 1570.8884268776492, 'y': 918.5637477298636, 'z': 120.93222187863965, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 3769.899023799315, 'y': 3401.160626592799, 'z': 187.75126628534275, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 4430.1001623689135, 'y': 2165.6623848067557, 'z': 95.9466625774489, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}, {'x': 661.5209040784191, 'y': 4464.76888456163, 'z': 238.56128572473233, 'v_x': 0.0, 'v_y': 0.0, 'v_z': 0.0}]
        self.dynamic_obs_init_state = [{'x': 3361.9312062379227, 'y': 1301.8970838259083, 'z': 258.096596783287, 'v_x': 46.431137634250405, 'v_y': 42.23479720836719, 'v_z': -0.33226113346565267}, {'x': 3201.3940965029815, 'y': 4044.4550255704903, 'z': 294.5524500656295, 'v_x': 8.600969627495658, 'v_y': 7.761353358646344, 'v_z': 2.2358511675594084}, {'x': 3542.3929425163515, 'y': 2390.304895993673, 'z': 111.37794779834601, 'v_x': 0.5793027378000304, 'v_y': 11.02524640567561, 'v_z': 0.6114454368066902}, {'x': 1272.3086931784344, 'y': 2755.656872221971, 'z': 7.547512397695222, 'v_x': 27.356323863467008, 'v_y': 43.6351433213272, 'v_z': 0.6287392923682038}, {'x': 3465.8752843790826, 'y': 1323.2357733711683, 'z': 79.80925656542647, 'v_x': 41.66880952568927, 'v_y': 45.189581307394825, 'v_z': -2.4197211925532525}, {'x': 827.3286060286069, 'y': 1798.2690843401056, 'z': 213.3632538617275, 'v_x': 13.675878937093488, 'v_y': 15.836100373261624, 'v_z': -2.969021810621121}, {'x': 673.8464351867694, 'y': 724.4988427269203, 'z': 101.27700729268115, 'v_x': 38.56658813882422, 'v_y': 22.53994938526807, 'v_z': -1.5958457123414682}, {'x': 708.512295691738, 'y': 3151.5448159608272, 'z': 68.95316209385169, 'v_x': 45.53113051526723, 'v_y': 16.08870528718836, 'v_z': 1.9083328282671}, {'x': 2877.870842835495, 'y': 3565.3894589977244, 'z': 254.79831950787943, 'v_x': 3.853128069904188, 'v_y': 31.363455570431093, 'v_z': -1.440230073682307}, {'x': 3548.6385688749156, 'y': 1034.649153582318, 'z': 257.9151207139955, 'v_x': 41.18218166240531, 'v_y': 20.867956009711502, 'v_z': 2.2449519414313617}]
        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        self.att = 2 / ((self.target[0]['x']/1000 - self.uav_init_state[0]['x']/1000) ** 2 +
                        (self.target[0]['y']/1000 - self.uav_init_state[0]['y']/1000) ** 2 +
                        (self.target[0]['z']/1000 - self.uav_init_state[0]['z']/1000) ** 2)
        self.rep = 2 / (1 / self.min_sep_hori - 1 / self.min_range) ** 2
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
        done = False
        reward = 0.0

        # pre_tar_dis =  np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
        #                   (self.uav_state[0]['y'] - self.target[0]['y']) ** 2 +
        #                   (self.uav_state[0]['z'] - self.target[0]['z']) ** 2)
        # Calculate the uav's position
        self.uav_state[0]['v_x'] += action[0]['delta_v_x']
        if self.uav_state[0]['v_x'] > 200:
            self.uav_state[0]['v_x'] = 200
            reward += -20
        elif self.uav_state[0]['v_x'] < -200:
            self.uav_state[0]['v_x'] = -200
            reward += -20
        self.uav_state[0]['x'] += self.uav_state[0]['v_x'] * self.delta_t
        self.uav_state[0]['v_y'] += action[0]['delta_v_y']
        if self.uav_state[0]['v_y'] > 200:
            self.uav_state[0]['v_y'] = 200
            reward += -20
        elif self.uav_state[0]['v_y'] < -200:
            self.uav_state[0]['v_y'] = -200
            reward += -20
        self.uav_state[0]['y'] += self.uav_state[0]['v_y'] * self.delta_t
        self.uav_state[0]['v_z'] += action[0]['delta_v_z']
        if self.uav_state[0]['v_z'] > 10:
            self.uav_state[0]['v_z'] = 10
            reward += -20
        elif self.uav_state[0]['v_z'] < -10:
            self.uav_state[0]['v_z'] = -10
            reward += -20
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
        static_obs_dis_vert = [self.static_obs_state[i]['z'] - self.uav_state[0]['z'] for i in range(len(self.static_obs_state))]
        dynamic_obs_dis = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2 +
            (self.dynamic_obs_state[i]['z'] - self.uav_state[0]['z']) ** 2) for i in range(len(self.dynamic_obs_state))]
        dynamic_obs_dis_hori = [np.sqrt(
            (self.dynamic_obs_state[i]['x'] - self.uav_state[0]['x']) ** 2 +
            (self.dynamic_obs_state[i]['y'] - self.uav_state[0]['y']) ** 2) for i in range(len(self.dynamic_obs_state))]
        dynamic_obs_dis_vert = [self.dynamic_obs_state[i]['z'] - self.uav_state[0]['z'] for i in range(len(self.dynamic_obs_state))]
        tar_dis = np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
                          (self.uav_state[0]['y'] - self.target[0]['y']) ** 2 +
                          (self.uav_state[0]['z'] - self.target[0]['z']) ** 2)
        tar_dis_hori = np.sqrt((self.uav_state[0]['x'] - self.target[0]['x']) ** 2 +
                          (self.uav_state[0]['y'] - self.target[0]['y']) ** 2)
        tar_dis_vert = self.uav_state[0]['z'] - self.target[0]['z']
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
        # uatt = np.exp(-tar_dis/1000)
        # reward += 20 * uatt - sum(urep_static) - sum(urep_dynamic)

        urep_static = np.where(np.array(static_obs_dis) < self.min_range,
                               - 1 / 2 * self.rep * (1 / np.array(static_obs_dis) - 1 / self.min_range), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
                                - 1 / 2 * self.rep * (1 / np.array(dynamic_obs_dis) - 1 / self.min_range), 0)
        uatt = - self.att * tar_dis
        reward += 0.05*sum(urep_static) + 0.05*sum(urep_dynamic) + 0.1*uatt


        # reward += -0.1
        # reward += (pre_tar_dis - tar_dis)/10

        if tar_dis_vert < self.min_sep_vert and tar_dis_hori < self.min_sep_hori:
            reward += 2000
            # print(" --------reach the goal")
            done = True
        elif np.any(np.array(np.abs(np.array(static_obs_dis_vert)) < self.min_sep_vert) &
                  np.array(np.abs(np.array(static_obs_dis_hori)) < self.min_sep_hori)) or \
                np.any(np.array(np.abs(np.array(dynamic_obs_dis_vert)) < self.min_sep_vert) &
                np.array(np.abs(np.array(dynamic_obs_dis_hori)) < self.min_sep_hori)):
            reward += -2000
            # print(" --------collision")
            done = True
        elif self.uav_state[0]['x'] < 0.0 or self.uav_state[0]['x'] > 5000.0 or \
            self.uav_state[0]['y'] < 0.0 or self.uav_state[0]['y'] > 5000.0 or \
                self.uav_state[0]['z'] < 0.0 or self.uav_state[0]['z'] > 300.0:
            reward += -2000
            # print(" --------out of the boundary")
            done = True
        elif len(self.path) > self._max_episode_steps:
            done = True

        return self.uav_state, reward, done

    def sample_action(self):
        # # Normal
        # random_delta_v_x = np.random.normal() * 2.0
        # random_delta_v_y = np.random.normal() * 2.0
        # random_delta_v_z = np.random.normal() * 0.3
        # # Mean
        random_delta_v_x = np.random.rand() * 10.0
        random_delta_v_y = np.random.rand() * 10.0
        random_delta_v_z = np.random.rand() * 1.0
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

    def plot_reward(self, x, y, z = 0):
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
        # uatt = np.exp(-tar_dis/1000)
        # reward += 20 * uatt - sum(urep_static) - sum(urep_dynamic)

        urep_static = np.where(np.array(static_obs_dis) < self.min_range,
                               - 1 / 2 * self.rep * (1 / np.array(static_obs_dis) - 1 / self.min_range), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
                               - 1 / 2 * self.rep * (1 / np.array(dynamic_obs_dis) - 1 / self.min_range), 0)
        uatt = - self.att * tar_dis
        reward += 0.05*sum(urep_static) + 0.05*sum(urep_dynamic)
        reward += 0.1*uatt

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

    # for i in range(50):
    #     env.render()
    #     time.sleep(0.1)
    #     action = env.sample_action()
    #     s, r, done = env.step(action)
    #     print(f"currently, the {i + 1} step:\n"
    #           f"           Action: speed {action[0]['delta_v_x'], action[0]['delta_v_y'], action[0]['delta_v_z']}\n"
    #           f"           State: pos {s[0]['x'], s[0]['y'], s[0]['z']};   speed {s[0]['v_x'], s[0]['v_y'], s[0]['v_z']}\n"
    #           f"           Reward:{r}\n")
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
