# coding=utf-8
import time
import pyglet
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import copy


class SingleContinuousEnv(object):
    def __init__(self, seed=0):
        self.env_kinds = 4
        self._max_episode_steps = 100
        self.model = None
        self.randomly_choose_env()
        self.seed(seed)


    def randomly_choose_env(self):
        # randomly choose an env
        self.model = SingleContinuousEnv0(self._max_episode_steps)

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
        self.model.show_path()

    def close(self):
        self.model.close()


class SingleContinuousEnv0(object):
    def __init__(self, max_episode_steps):
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
        self.min_sep = 50
        self.min_range = 500
        self.viewer = None
        self._max_episode_steps = max_episode_steps

        self.init_map()

    def init_map(self):
        self.target = [{'x': 5000, 'y': 5000, 'z': 300}]
        self.uav_init_state = [{'x': 0, 'y': 0, 'z': 0, 'v_hori': 100 * np.sqrt(2), 'v_vert': 6, 'angle_hori': (1 / 4) * np.pi}]
        self.static_obs_state = [
            {'x': 2467.740753445062, 'y': 3633.2198775704355, 'z': 140.39133597414653, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 4818.559080490401, 'y': 146.282746499829, 'z': 229.10676961871667, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 1522.484011349053, 'y': 1791.5075907216494, 'z': 292.21182855577604, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 3513.901665772854, 'y': 3495.921898679873, 'z': 272.1442077516158, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 1004.3273407574554, 'y': 2820.6396264390605, 'z': 144.94763111852643, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 3190.360967144129, 'y': 4090.848854639964, 'z': 240.91046370756337, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 4703.383238174666, 'y': 1458.450990816234, 'z': 116.14210320486677, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 1143.5881325136127, 'y': 1770.9504222709134, 'z': 198.26479363235364, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 2412.5687119819613, 'y': 3590.735277003564, 'z': 89.1342727522315, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 1749.6619023841974, 'y': 4334.557608492229, 'z': 226.6584770671501, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 4168.8811962785685, 'y': 709.9672976301724, 'z': 266.399731684341, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 107.69766901818744, 'y': 618.5399949745369, 'z': 24.05383456665272, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 4308.934406856321, 'y': 2048.7766401716935, 'z': 209.28780028870486, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 3113.1233074346246, 'y': 1791.6450133233707, 'z': 271.92287916735745, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 4308.70981457602, 'y': 3816.911019609893, 'z': 227.70993827506004, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 1.9605973819414313, 'y': 2702.3212999420766, 'z': 9.232720683933094, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 622.0392255338097, 'y': 2471.4632632245502, 'z': 91.58423784513788, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 4466.402072706533, 'y': 4031.0650850085426, 'z': 187.45491155403496, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 1774.8230191296066, 'y': 4228.417831474892, 'z': 10.677569869485936, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0},
            {'x': 311.19649900346155, 'y': 4425.48609123886, 'z': 161.9521919345718, 'v_hori': 0.0, 'v_vert': 0.0,
             'angle_hori': 0.0}]
        self.dynamic_obs_init_state = [{'x': 2328.631383806805, 'y': 1138.5528822762847, 'z': 197.86861278111917,
                                        'v_hori': 21.12829419147405, 'v_vert': -0.2282255706010341,
                                        'angle_hori': 1.6684623975523836},
                                       {'x': 4705.600184614786, 'y': 2387.995360834459, 'z': 99.78453203312725,
                                        'v_hori': 36.448001124744586, 'v_vert': -1.1689669220550287,
                                        'angle_hori': 2.825910777652289},
                                       {'x': 4198.836218178802, 'y': 1798.6095184529988, 'z': 4.5023170720720485,
                                        'v_hori': 40.48215310838126, 'v_vert': -1.5899877388590182,
                                        'angle_hori': 0.9761917611229763},
                                       {'x': 3238.6296956897227, 'y': 2632.271355710324, 'z': 283.27079944710056,
                                        'v_hori': 48.94283585702374, 'v_vert': 0.097744538492881,
                                        'angle_hori': 4.7002757181323656},
                                       {'x': 2270.6786222481924, 'y': 2454.0827936218775, 'z': 178.59215486973991,
                                        'v_hori': 39.67523023153737, 'v_vert': 2.3340963451535783,
                                        'angle_hori': 4.448171869629758}]

        self.uav_state = copy.deepcopy(self.uav_init_state)
        self.dynamic_obs_state = copy.deepcopy(self.dynamic_obs_init_state)
        self.att = 2 / ((self.target[0]['x'] - self.uav_init_state[0]['x']) ** 2 + (
                    self.target[0]['y'] - self.uav_init_state[0]['y']) ** 2)
        self.rep = 2 / (1 / self.min_sep - 1 / self.min_range) ** 2
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
        # Calculate the uav's position
        self.uav_state[0]['angle_hori'] += action[0]['delta_angle']
        delta_xy_dis = self.uav_state[0]['v_hori'] * self.delta_t + (1 / 2) * action[0]['a_hori'] * (self.delta_t ** 2)
        delta_z_dis = self.uav_state[0]['v_vert'] * self.delta_t + (1 / 2) * action[0]['a_vert'] * (self.delta_t ** 2)
        self.uav_state[0]['x'] += delta_xy_dis * np.cos(self.uav_state[0]['angle_hori'])
        self.uav_state[0]['y'] += delta_xy_dis * np.sin(self.uav_state[0]['angle_hori'])
        self.uav_state[0]['z'] += delta_z_dis
        self.uav_state[0]['v_hori'] += action[0]['a_hori'] * self.delta_t
        self.uav_state[0]['v_vert'] += action[0]['a_vert'] * self.delta_t
        self.save_path([self.uav_state[0]['x'], self.uav_state[0]['y'], self.uav_state[0]['z']])
        # Calculate the dynamic obstacles' position
        for obs in self.dynamic_obs_state:
            obs_delta_xy_dis = obs['v_hori'] * self.delta_t
            obs_delta_z_dis = obs['v_vert'] * self.delta_t
            obs['x'] += obs_delta_xy_dis * np.cos(obs['angle_hori'])
            obs['y'] += obs_delta_xy_dis * np.sin(obs['angle_hori'])
            obs['z'] += obs_delta_z_dis
        # Calculate the reward
        done = False
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
        urep_static = np.where(np.array(static_obs_dis) < self.min_range,
                               - 1 / 2 * self.rep * ((1 / np.array(static_obs_dis) - 1 / self.min_range) ** 2), 0)
        urep_dynamic = np.where(np.array(dynamic_obs_dis) < self.min_range,
                                - 1 / 2 * self.rep * ((1 / np.array(dynamic_obs_dis) - 1 / self.min_range) ** 2), 0)
        uatt = - 1 / 2 * self.att * (tar_dis ** 2)
        reward = sum(urep_static) * 20 + sum(urep_dynamic) * 20 + 1000 * uatt

        if tar_dis < self.min_sep:
            reward += 1000
            done = True
        if np.any(np.array(static_obs_dis) < self.min_sep) or np.any(np.array(dynamic_obs_dis) < self.min_sep):
            reward += -500
            done = True
        if self.uav_state[0]['x'] > 6000 or self.uav_state[0]['x'] < -1000 or self.uav_state[0]['y'] > 6000 or self.uav_state[0]['y'] < -1000 or self.uav_state[0]['z'] > 360 or self.uav_state[0]['z'] < -60:
            done = True
        if len(self.path) > self._max_episode_steps:
            done = True

        return self.uav_state, reward, done

    def sample_action(self):
        # Normal
        # random_a_hori = np.random.normal(loc=0.0, scale=1.2)
        # random_a_vert = np.random.normal(loc=0.0, scale=0.012)
        # random_delta_angle = np.random.normal(loc=0.0, scale=1.2) / 180 * np.pi
        random_a_hori = np.random.normal() * 6
        random_a_vert = np.random.normal() * 0.06
        random_delta_angle = np.random.normal() * 6 / 180 * np.pi
        # random_a_hori = np.random.normal() * 1.2
        # random_a_vert = np.random.normal() * 0.012
        # random_delta_angle = np.random.normal() * 1.2 / 180 * np.pi
        return [{'a_hori': random_a_hori, 'a_vert': random_a_vert, 'delta_angle': random_delta_angle}]

    def save_path(self, next_state):
        self.path.append(next_state)

    def show_path(self):
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
        ax.scatter([self.path[path_len-1][0] / 1000], [self.path[path_len-1][1] / 1000],
                   [self.path[path_len-1][2] / 1000], color='green', alpha=0.7, label='UAV')
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

        ax.set_title("path")
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


class Viewer(pyglet.window.Window):
    def __init__(self, uav_state, target, static_obs_state, dynamic_obs_state):
        super(Viewer, self).__init__(width=500, height=500, resizable=False, caption='single-agent',
                                     vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.uav_state = uav_state
        self.target = target
        self.uav = pyglet.shapes.Circle(self.uav_state[0]['x']/10, self.uav_state[0]['y']/10, 3,
                                        color=(1, 100, 1))
        self.target = pyglet.shapes.Circle(self.target[0]['x']/10, self.target[0]['y']/10, 3, color=(1, 200, 1))
        self.static_obs = []
        self.static_obs_state = static_obs_state
        for j in range(len(static_obs_state)):
            cur_obs = pyglet.shapes.Circle(self.static_obs_state[j]['x']/10, self.static_obs_state[j]['y']/10, 3,
                                           color=(86, 109, 249))
            self.static_obs.append(cur_obs)
        self.dynamic_obs = []
        self.dynamic_obs_state = dynamic_obs_state
        for j in range(len(dynamic_obs_state)):
            cur_obs = pyglet.shapes.Circle(self.dynamic_obs_state[j]['x']/10, self.dynamic_obs_state[j]['y']/10, 3,
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
        self.uav = pyglet.shapes.Circle(self.uav_state[0]['x']/10, self.uav_state[0]['y']/10, 3,
                                        color=(1, 100, 1))
        temp = []
        for j in range(len(self.dynamic_obs_state)):
            cur_obs = pyglet.shapes.Circle(self.dynamic_obs_state[j]['x']/10, self.dynamic_obs_state[j]['y']/10, 3,
                                           color=(249, 109, 249))
            temp.append(cur_obs)
        self.dynamic_obs = temp


if __name__ == "__main__":
    env = SingleContinuousEnv()
    env.reset()

    for i in range(50):
        env.render()
        time.sleep(0.02)
        action = env.sample_action()
        # print(f"horizontal a: {action[0]['a_hori']}, vertical a: {action[0]['a_vert']}, delta angle: {action[0]['delta_angle']/np.pi*180}")
        s, r, done = env.step(action)
        print(f"currently, the {i + 1} step:")
        print(f"           Action: speed {action[0]['a_hori'], action[0]['a_vert']};   angle {action[0]['delta_angle'] / np.pi * 180}\n"
              f"           State: pos {s[0]['x'], s[0]['y'], s[0]['z']};   speed {s[0]['v_hori'], s[0]['v_vert']};   angle {s[0]['angle_hori'] / np.pi * 180}\n"
              f"           Reward:{r}")
        if done:
            break

    env.show_path()
