# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter


def create_map():
    static_obstacle_num = 100
    dynamic_obstacle_num = 5
    static_obs_info = []
    dynamic_obs_info = []
    # static_obs_info = np.empty(
    #     static_obstacle_num,
    #     dtype={'x': np.float32, 'y': np.float32, 'z': np.float32, 'v_hori': np.float32, 'v_vert': np.float32,
    #            'angle_hori': np.float32})
    # dynamic_obs_info = np.empty(
    #     dynamic_obstacle_num,
    #     dtype={'x': np.float32, 'y': np.float32, 'z': np.float32, 'v_hori': np.float32, 'v_vert': np.float32,
    #            'angle_hori': np.float32})
    for i in range(static_obstacle_num):
        x_i = np.random.rand(1) * 5000
        y_i = np.random.rand(1) * 5000
        z_i = np.random.rand(1) * 300
        v_hori_i = 0
        v_vert_i = 0
        angle_hori_i = 0
        static_obs_info.append({'x': float(x_i), 'y': float(y_i), 'z': float(z_i),
                                'v_hori': float(v_hori_i), 'v_vert': float(v_vert_i), 'angle_hori': float(angle_hori_i)})
    for i in range(dynamic_obstacle_num):
        # theta_i = np.random.rand(1)
        # r_i = np.random.rand(1) * (40 - 9.26) +9.26
        # x_i = np.cos(theta_i) * r_i
        # y_i = np.sin(theta_i) * r_i
        # z_i = np.random.rand(1) * 3 - 1.5
        x_i = np.random.rand(1) * 5000
        y_i = np.random.rand(1) * 5000
        z_i = np.random.rand(1) * 300
        v_hori_i = np.random.rand(1) * 50
        v_vert_i = np.random.rand(1) * 6 - 3
        angle_hori_i = np.random.rand(1) * np.pi * 2
        dynamic_obs_info.append({'x': float(x_i), 'y': float(y_i), 'z': float(z_i),
                                'v_hori': float(v_hori_i), 'v_vert': float(v_vert_i), 'angle_hori': float(angle_hori_i)})
    map = np.concatenate((static_obs_info, dynamic_obs_info), axis=0)
    print(static_obs_info)
    print(dynamic_obs_info)
    # print(map)
    return static_obs_info, dynamic_obs_info


def plot_test():
    map = create_map()
    map_len = len(map)
    xscatter = [map[i][0] / 1000 for i in range(map_len)]
    yscatter = [map[i][1] / 1000 for i in range(map_len)]
    zscatter = [map[i][2] / 1000 for i in range(map_len)]

    path = [[0, 0, 0], [1699.4557, 2794.4287, 202.93611], [4452.5464, 3547.4946, 231.30966]]
    x = [path[i][0] / 1000 for i in range(len(path))]
    y = [path[i][1] / 1000 for i in range(len(path))]
    z = [path[i][2] / 1000 for i in range(len(path))]

    ax = plt.axes(projection='3d')  # or use :    fig = plt.figure()   ax = fig.add_subplot(projection="3d")
    ax.scatter(xscatter, yscatter, zscatter, label='obstacle', c='r', alpha=0.7)
    ax.plot3D(x, y, z, label='path')
    ax.set_title("path")
    ax.set_xlabel("x (km)")
    ax.set_xlim(0, 5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylabel("y (km)")
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_zlabel("z (km)")
    ax.set_zlim(0, 0.3)
    ax.zaxis.set_major_locator(MultipleLocator(0.05))

    ax.legend()
    plt.show()


def save_to_file(data, file_name):
    # https://www.cnblogs.com/tester-hqser/p/16202786.html
    pass


if __name__ == "__main__":
    static_obs_info, dynamic_obs_info = create_map() # 测试
    # save_to_file(static_obs_info, "static_obs")
    # save_to_file(dynamic_obs_info, "dynamic_obs")
    print(np.array(list(static_obs_info[0].values())))
    print(np.array([list(static_obs_info[i].values()) for i in range(len(static_obs_info))]))
