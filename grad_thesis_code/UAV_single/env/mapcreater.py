# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

def create_map():
    static_obstacle_num = 10
    dynamic_obstacle_num = 40
    static_obs_info = []
    dynamic_obs_info = []
    for i in range(static_obstacle_num):
        x_i = np.random.rand() * 4000 + 500
        y_i = np.random.rand() * 4000 + 500
        z_i = np.random.rand() * 300
        v_x_i = 0.
        v_y_i = 0.
        v_z_i = 0.
        static_obs_info.append({'x': x_i, 'y': y_i, 'z': z_i,
                                'v_x': v_x_i, 'v_y': v_y_i, 'v_z': v_z_i})
    for i in range(dynamic_obstacle_num):
        x_i = np.random.rand() * 4000 + 500
        y_i = np.random.rand() * 4000 + 500
        z_i = np.random.rand() * 300
        v_x_i = np.random.rand() * 50 - 25
        v_y_i = np.random.rand() * 50 - 25
        v_z_i = np.random.rand() * 3 - 1.5
        dynamic_obs_info.append({'x': x_i, 'y': y_i, 'z': z_i,
                                 'v_x': v_x_i, 'v_y': v_y_i, 'v_z': v_z_i})
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


if __name__ == "__main__":
    create_map()
