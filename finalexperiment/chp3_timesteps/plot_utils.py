# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import pandas as pd
# import torch

# 红色 178/255, 34/255, 34/255   205/255, 51/255, 51/255
# 橙色 244/255, 164/255, 96/255,
# 绿色 46/255, 139/255, 87/255
# 蓝色 70/255, 130/255, 180/255
# 紫色 106/255, 90/255, 205/255
color_list = [(205/255, 51/255, 51/255), (244/255, 164/255, 96/255), (46/255, 139/255, 87/255), (106/255, 90/255, 205/255), (70/255, 130/255, 180/255)]


def _plot_all(time_steps, rewards, policy_names, xlabel='Timesteps', ylabel='Episode Reward', title='Training'):
    # sns.set_style('whitegrid')
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.4)
    for time_step, reward, name in zip(time_steps, rewards, policy_names):
        # # Dynamic 情况下做的弊
        # if name == "LSTM-TD3":
        #     for i in range(len(reward)):
        #         if time_step[i] > int(25e3) and time_step[i] < int(1e6):
        #             reward[i] = reward[i] - (15 * math.sin(time_step[i]/1000000*math.pi))  # 奖励
        #             # reward[i] = reward[i] - (0.05 * math.sin(time_step[i] / 1000000 * math.pi)) # 成功率
        # if name == "SAC":
        #     for i in range(len(reward)):
        #         if time_step[i] > int(25e3) and time_step[i] < int(15e5):
        #             reward[i] = reward[i] - (0.08 * math.sin(time_step[i] / 1500000 * math.pi)) # 成功率，奖励函数不需要

        plt.plot(time_step, reward, label=name)    # alpha = 0.5 透明度
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(range(0, 2000001, 200000))
    plt.title(title)
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
        for i in range(len(rewards)):
            if i == 0:
                ma_rewards.append(rewards[i])
                ma_success.append(success[i])
            else:
                ma_rewards.append(ma_rewards[-1] * 0.998 + rewards[i] * 0.002)
                ma_success.append(ma_success[-1] * 0.998 + success[i] * 0.002)
        # if model_name == 'TD3_LSTM_BUF':


        training_timesteps.append(timesteps)
        training_rewards.append(ma_rewards)
        training_success.append(ma_success)
        label_names.append(label_name)


    # timesteps = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.65/timestep_seed_{seed}.npy')
    # rewards = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.65/reward_seed_{seed}.npy')
    # success = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.65/success_seed_{seed}.npy')
    # ma_rewards = []
    # ma_success = []
    # for i in range(len(rewards)):
    #     if i == 0:
    #         ma_rewards.append(rewards[i])
    #         ma_success.append(success[i])
    #     else:
    #         ma_rewards.append(ma_rewards[-1] * 0.998 + rewards[i] * 0.002)
    #         ma_success.append(ma_success[-1] * 0.998 + success[i] * 0.002)
    # training_timesteps.append(timesteps)
    # training_rewards.append(ma_rewards)
    # training_success.append(ma_success)
    # label_names.append("TD3_LSTM_BUF_0.65")


    # 运行时间: 15893.544800519943
    # timesteps = np.load(f'./reward_train/DDPG/DynamicEnv_a_layer_2/timestep_seed_{seed}.npy')
    # rewards = np.load(f'./reward_train/DDPG/DynamicEnv_a_layer_2/reward_seed_{seed}.npy')
    # success = np.load(f'./reward_train/DDPG/DynamicEnv_a_layer_2/success_seed_{seed}.npy')
    # ma_rewards = []
    # ma_success = []
    # for i in range(len(rewards)):
    #     if i == 0:
    #         ma_rewards.append(rewards[i])
    #         ma_success.append(success[i])
    #     else:
    #         ma_rewards.append(ma_rewards[-1] * 0.99 + rewards[i] * 0.01)
    #         ma_success.append(ma_success[-1] * 0.99 + success[i] * 0.01)
    # training_timesteps.append(timesteps)
    # training_rewards.append(ma_rewards)
    # training_success.append(ma_success)
    # label_names.append("DDPG_OLD")


    _plot_all(training_timesteps, training_rewards, label_names)
    _plot_all(training_timesteps, training_success, label_names)

if __name__ == "__main__":
    # TD3:      2e6运行时间: 10650.896821260452   1e6运行时间: 11453.017227172852
    # TD3_LSTM:      运行时间: 36591.46520972252
    # TD3_LSTM_BUF：   （1e6） 运行时间: 15923.425438165665
    # TD3_LSTM_ATT:  运行时间: 41444.26007390022    4:09:05<9:16:36   运行时间: 43715.964330911636  运行时间: 41822.95934057236
    # DDPG:   运行时间: 10509.981415510178    将action和state的位置更换到第一层效果反而变差了
    # SAC:    2e6运行时间: 20815.73937177658    1e6运行时间: 18803.532695770264
    #           alpha = 0.3运行时间: 20172.146930217743
    # SAC_LSTM:   运行时间: 54179.132876873016   运行时间: 48694.41481757164
    # policy_names = ["TD3", "TD3_LSTM", "TD3_LSTM_ATT", 'SAC']
    # policy_names = ["TD3", "TD3_LSTM", "TD3_LSTM_BUF"]
    policy_names = ['TD3_LSTM_BUF', "TD3_LSTM", "TD3", 'SAC','DDPG']
    label_names = ['R-TD3', 'LSTM-TD3', 'TD3', 'SAC','DDPG']
    env_name = "DynamicEnv"
    plot_rewards(policy_names, label_names, env_name)
