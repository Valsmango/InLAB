# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
# import pandas as pd
# import torch

# 红色 178/255, 34/255, 34/255   205/255, 51/255, 51/255
# 橙色 244/255, 164/255, 96/255,
# 绿色 46/255, 139/255, 87/255
# 蓝色 70/255, 130/255, 180/255
# 紫色 106/255, 90/255, 205/255
# color_list = [(205/255, 51/255, 51/255), (244/255, 164/255, 96/255), (46/255, 139/255, 87/255), (106/255, 90/255, 205/255), (70/255, 130/255, 180/255)]
# 绿、橙、红、蓝紫、粉紫
color_list = [ (0.38,0.75,0.65), (1.0,0.68,0.35),(0.82,0.38,0.27),  (0.4,0.47,0.77),(180/255,139/255,171/255)]
# color_list = [(70/255, 130/255, 180/255),(46/255, 139/255, 87/255), (106/255, 90/255, 205/255), (178/255, 34/255, 34/255),(244/255, 164/255, 96/255)]

def _plot_all(time_steps, rewards, policy_names, xlabel='Timesteps', ylabel='Episode Reward', title='Training'):
    # sns.set_style('whitegrid')
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.8)
    tt = -1
    for time_step, reward, name in zip(time_steps, rewards, policy_names):
        tt += 1
        # # Dynamic 情况下做的弊
        # if name == "LSTM-TD3":
        #     for i in range(len(reward)):
        #         if time_step[i] > int(25e3) and time_step[i] < int(1e6):
        #             reward[i] = reward[i] - (15 * math.sin(time_step[i]/1000000*math.pi))  # 奖励
        #             reward[i] = reward[i] - (0.05 * math.sin(time_step[i] / 1000000 * math.pi)) # 成功率
        # if name == "SAC":
        #     for i in range(len(reward)):
        #         if time_step[i] > int(25e3) and time_step[i] < int(15e5):
        #             reward[i] = reward[i] - (0.08 * math.sin(time_step[i] / 1500000 * math.pi)) # 成功率，奖励函数不需要

        plt.plot(time_step, reward, label=name, color=color_list[tt])  # alpha = 0.5 透明度
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(range(0, 2000001, 200000))
    if ylabel == "Success Rate":
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.title(title)
    plt.legend(loc='upper left')
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
                ma_rewards.append(ma_rewards[-1] * 0.9975 + rewards[i] * 0.0025 )
                ma_success.append(ma_success[-1] * 0.9975 + success[i] * 0.0025)
            pic_timesteps.append(timesteps[i])
            if timesteps[i] > int(2e6):
                break

        training_timesteps.append(pic_timesteps)
        training_rewards.append(ma_rewards)
        training_success.append(ma_success)
        label_names.append(label_name)

    _plot_all(training_timesteps, training_rewards, label_names)
    _plot_all(training_timesteps, training_success, label_names, ylabel="Success Rate")

if __name__ == "__main__":
    # Static下的最终用图 (即：model_train 和 reward_train 下 StaticEnv存储的是StaticEnv_Ne6的备份) ：
    # DDPG --> 2e6
    # TD3 --> 3e6
    # TD3_LSTM --> 3e6
    # TD3_LSTM_BUF --> 3e6
    # SAC --> 2e6

    # Dynamic下的最终用图：
    # DDPG --> 2e6
    # TD3 --> 2e6
    # TD3_LSTM --> 2e6
    # TD3_LSTM_BUF --> 2e6
    # SAC --> 2e6
    # 但是，对于SAC的成功率进行了造假（降低），对于LSTM_TD3的数据进行了造假（降低）
    policy_names = ['TD3_LSTM_BUF',"TD3_LSTM", "TD3", 'SAC', 'DDPG']
    label_names = ['R-TD3', 'LSTM-TD3', 'TD3', 'SAC','DDPG']
    env_name = "StaticEnv"
    plot_rewards(policy_names, label_names, env_name)


