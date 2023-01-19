import time
import pyglet  # python的可视化模块，gym里面的游戏场景也是基于这个的
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

'''
gym的env包括：
    env.reset() 重新开局
    env.render() 可视化
    env.step()  返回 next_state, reward, done
整个游戏区域：
    500 x 500，划分为50格，一格为 10 pixels（也就是以10m/s的速度匀速飞行，走大约50步左右就可以到达终点）
无人机的state和action设置：
    state: 4 个维度，x轴坐标，y轴坐标，角度（与x轴夹角），水平速度; 设定航向角为0 - 360度，即np.pi*2; 设定速度为 0 - 20 m/s
    action： 2个维度，角度增量，水平速度增量
    max_action： 每一步所选动作的最大值，最大角度增量设为 +-5 度（np.pi/36），速度增加量最大为 +-1 m/s

可以修改的点：
    1，场景设置（游戏的区域、障碍物位置、max_action、初始速度、每一步走多远等等）；pyglet是否必须？可以换成matplotlib绘图吗？
    2，和算法模型相关的关键信息：奖励设置，r大小为多少更能体现出DDPG模型比随机sample_action更好？奖励是基于简化的人工势场法进行设置的，还有其他约束条件可以加进来吗？
    3，sample_action的设置后续可以改为正态分布，再进行效果对比

可能存在bug的地方：
    seed的设置原理还没细看
'''


class singleEnv(object):
    target = {'x': 495., 'y': 495.}
    action_dim = 2  # 角度(与x轴夹角)和速度
    state_dim = 4  # 位置[x, y]、角度和速度
    dt = 1  # 一秒一步
    v_bound = [1, 20]  # 最快20m/s, angle_bound不需要bound是因为可以直接 % 360度
    viewer = None

    def __init__(self):
        self.uav_info = np.zeros(
            1, dtype=[('x', np.float32), ('y', np.float32), ('v', np.float32), ('angle', np.float32)])  # 只有一架飞行器
        self.obs_info = np.zeros(
            4, dtype=[('x', np.float32), ('y', np.float32)])  # 生成4个障碍物
        self.seed()
        self.reset()
        self.att = 2 / ((self.target['x'] - self.uav_info['x']) ** 2 + (self.target['y'] - self.uav_info['y']) ** 2)
        self.min_range = 100  # 假设斥力场的作用范围为100m
        self.min_sep = 10  # 假设10m以内就算撞上了
        self.rep = 2 / (1 / self.min_sep - 1 / self.min_range) ** 2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):  # 没有加入噪声，也就是每一次重新开始都是相同的场景
        self.uav_info['x'] = 5
        self.uav_info['y'] = 5
        self.uav_info['v'] = 10
        self.uav_info['angle'] = np.pi / 4
        self.obs_info['x'] = [105, 205, 305, 405]
        self.obs_info['y'] = [105, 205, 305, 405]
        s = self.uav_info['x']
        s = np.append(s, self.uav_info['y'])
        s = np.append(s, self.uav_info['v'])
        s = np.append(s, self.uav_info['angle'])
        return s

    def render(self):  # 调用Viewer中的render功能
        if self.viewer is None:
            self.viewer = Viewer(self.uav_info, self.target)
        self.viewer.render()

    def step(self, action):
        done = False
        r = 0.

        # uav根据action进行位移
        self.uav_info['v'] += action[0]
        self.uav_info['v'] = np.clip(self.uav_info['v'], *self.v_bound)
        self.uav_info['angle'] += action[1]
        self.uav_info['angle'] %= np.pi * 2  # 设定航向角为0 - 360度
        self.uav_info['x'] += self.uav_info['v'] * np.cos(self.uav_info['angle']) * self.dt
        self.uav_info['y'] += self.uav_info['v'] * np.sin(self.uav_info['angle']) * self.dt

        s = self.uav_info['x']
        s = np.append(s, self.uav_info['y'])
        s = np.append(s, self.uav_info['v'])
        s = np.append(s, self.uav_info['angle'])

        # 计算势场：
        obs_dis = np.sqrt(
            (self.obs_info['x'] - self.uav_info['x']) ** 2 + (self.obs_info['y'] - self.uav_info['y']) ** 2)
        tar_dis = np.sqrt((self.uav_info['x'] - self.target['x']) ** 2 + (self.uav_info['y'] - self.target['y']) ** 2)
        urep = np.where(obs_dis < self.min_range, - 1 / 2 * self.rep * ((1 / obs_dis - 1 / self.min_range) ** 2), 0)
        uatt = - 1 / 2 * self.att * (tar_dis ** 2)
        r = sum(urep) + 10 * uatt

        # 达到目标点或者到达规划区域边界的奖励：
        if tar_dis < self.min_sep:
            r += 400
            done = True
        if np.any(obs_dis < self.min_sep):
            r += -200
            done = True
        return s, r, done

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def sample_action(self):
        # action：速度增量，航向偏移量
        action = np.random.rand(2)      # 均匀分布，而非正态分布，后续尝试修改
        action[0] = action[0] * 2 - 1       # 速度增量最大为 +-1 m/s
        action[1] = action[1] * np.pi / 18 - np.pi / 36  # 最大角度增量设为 +-5 度
        return action


class Viewer(pyglet.window.Window):
    def __init__(self, uav_info, target):  # 画出UAV和障碍物
        super(Viewer, self).__init__(width=500, height=500, resizable=False, caption='single-agent',
                                     vsync=False)  # 继承window模块
        pyglet.gl.glClearColor(1, 1, 1, 1)  # 设置白色背景
        self.uav_info = uav_info
        self.target = target
        self.uav = pyglet.shapes.Circle(self.uav_info['x'], self.uav_info['y'], 5,
                                        color=(1, 100, 1))  # 假设有50格，一格为10 pixel
        self.target = pyglet.shapes.Circle(self.target['x'], self.target['y'], 5, color=(1, 200, 1))
        self.obstacle_1 = pyglet.shapes.Circle(105, 105, 5, color=(86, 109, 249))
        self.obstacle_2 = pyglet.shapes.Circle(205, 205, 5, color=(86, 109, 249))
        self.obstacle_3 = pyglet.shapes.Circle(305, 305, 5, color=(86, 109, 249))
        self.obstacle_4 = pyglet.shapes.Circle(405, 405, 5, color=(86, 109, 249))

    def render(self):  # 刷新并呈现在屏幕上，基于pyglet，后续可以尝试简化为matplotlib，而不用pyglet
        self._update_uav()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):  # pyglet 是一个实时刷新的做动画模块, 所以每次刷新的时候, 会调用一个功能, on_draw() 就是 pyglet 刷新时需要的一个功能
        self.clear()  # 清屏
        self.uav.draw()
        self.target.draw()
        self.obstacle_1.draw()
        self.obstacle_2.draw()
        self.obstacle_3.draw()
        self.obstacle_4.draw()

    def _update_uav(self):  # 更新位置信息
        self.uav = pyglet.shapes.Circle(self.uav_info['x'], self.uav_info['y'], 5,
                                        color=(1, 100, 1))  # 假设有50格，一格为10 pixels


if __name__ == "__main__":
    env = singleEnv()
    env.reset()
    reward_info = np.zeros((1, 3))

    for i in range(100):
        env.render()
        time.sleep(0.01)  # 暂停一下，方便看清楚每一步
        action = env.sample_action()
        s, r, done = env.step(action)  # 用sample_action()来测试当前env设置的合理性

        reward_info = np.append(reward_info, [[s[0], s[1], r[0]]], axis=0)
        print(f"当前第{i + 1}步的位置为{s}")    # 打印具体的值，来测试角度增量、速度增量数值的设置是否可行
        print(f"           Action:速度增量{action[0]}；  角度增量{action[1] / np.pi * 180}\n"
              f"           State:位置{s[0], s[1]}；  速度{s[2]}；  角度{s[3] / np.pi * 180}\n"
              f"           Reward:{r}")
        if done:
            break
