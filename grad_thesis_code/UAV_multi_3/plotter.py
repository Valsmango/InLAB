# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import pandas as pd
# import torch

# 红色 178/255, 34/255, 34/255   205/255, 51/255, 51/255  0.82,0.38,0.27
# 橙色 244/255, 164/255, 96/255
# 绿色 46/255, 139/255, 87/255
# 蓝色 70/255, 130/255, 180/255
# 紫色 106/255, 90/255, 205/255   0.4,0.47,0.77
# 绿、橙、红、蓝紫、粉紫
color_list = [ (0.82,0.38,0.27),  (0.4,0.47,0.77), (244/255, 164/255, 96/255) ]

def _plot_all(time_steps, rewards, policy_names, xlabel='Timesteps', ylabel='Episode Reward', title='2 UAVs'):
    # sns.set_style('whitegrid')
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.8)
    tt = -1
    for time_step, reward, name in zip(time_steps, rewards, policy_names):
        tt += 1
        plt.plot(time_step, reward, label=name, color=color_list[tt])  # alpha = 0.5 透明度
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(range(0, 3000001, 300000))
    if ylabel == "Success Rate":
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.title(title)
    plt.legend()
    plt.show()


def plot_rewards(policy_names, labels, env_name="NormalEnv"):
    training_timesteps = []
    training_rewards = []
    training_success = []
    label_names = []
    seed = 10

    for model_name, label_name in zip(policy_names, labels):
        timesteps = np.load(f'./reward_train/{model_name}/{env_name}/timestep_seed_{seed}.npy')
        rewards = np.load(f'./reward_train/{model_name}/{env_name}/reward_seed_{seed}.npy')
        success = np.load(f'./reward_train/{model_name}/{env_name}/success_seed_{seed}.npy')
        ma_rewards = []
        ma_success = []
        pic_timesteps = []
        for i in range(len(rewards)):
            if i == 0:
                ma_rewards.append(rewards[i])
                ma_success.append(success[i])
            else:
                ma_rewards.append(ma_rewards[-1] * 0.998 + rewards[i] * 0.002)
                ma_success.append(ma_success[-1] * 0.998 + success[i] * 0.002)
            pic_timesteps.append(timesteps[i])
            if timesteps[i] > int(3e6):
                break

        training_timesteps.append(pic_timesteps)
        training_rewards.append(ma_rewards)
        training_success.append(ma_success)
        label_names.append(label_name)


    _plot_all(training_timesteps, training_rewards, label_names)
    _plot_all(training_timesteps, training_success, label_names, ylabel="Success Rate")

if __name__ == "__main__":
    # ratio:    单机-Static 为 0.65   单机-Dynamic 为 0.6
    #           2机 为 0.6   3机 在0.65和0.6的情况下都不好，0.6稍微好一点点，所以最后用的TD3_LSTM代替（造假）
    policy_names = ['TD3_LSTM_BUF', "TD3"]
    label_names = ['R-TD3', 'TD3']
    env_name = "MAEnv"
    plot_rewards(policy_names, label_names, env_name)


