# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch

# 红色 178/255, 34/255, 34/255   205/255, 51/255, 51/255
# 橙色 244/255, 164/255, 96/255,
# 绿色 46/255, 139/255, 87/255
# 蓝色 70/255, 130/255, 180/255
# 紫色 106/255, 90/255, 205/255
color_list = [(205/255, 51/255, 51/255), (244/255, 164/255, 96/255), (46/255, 139/255, 87/255), (106/255, 90/255, 205/255), (70/255, 130/255, 180/255)]


def _plot_all(time_steps, rewards, policy_names, xlabel='Timesteps', ylabel='Episode Reward', title='Training'):
    sns.set_style('whitegrid')
    for time_step, reward, name in zip(time_steps, rewards, policy_names):
        plt.plot(time_step, reward, label=name)    # alpha = 0.5 透明度
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(range(0, 2000001, 200000))
    plt.title(title)
    plt.legend()
    plt.show()

def plot_rewards(policy_names, env_name="NormalEnv"):
    training_timesteps = []
    training_rewards = []
    training_success = []
    label_names = []
    seed = 10

    for model_name in policy_names:
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
                ma_rewards.append(ma_rewards[-1] * 0.995 + rewards[i] * 0.005)
                ma_success.append(ma_success[-1] * 0.995 + success[i] * 0.005)

        training_timesteps.append(timesteps)
        training_rewards.append(ma_rewards)
        training_success.append(ma_success)
        label_names.append(model_name)


    # timesteps = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.5_0.5/timestep_seed_{seed}.npy')
    # rewards = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.5_0.5/reward_seed_{seed}.npy')
    # success = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.5_0.5/success_seed_{seed}.npy')
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
    # label_names.append("TD3_LSTM_BUF_0.5")
    # # 运行时间: 15893.544800519943
    # timesteps = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.9_0.1/timestep_seed_{seed}.npy')
    # rewards = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.9_0.1/reward_seed_{seed}.npy')
    # success = np.load(f'./reward_train/TD3_LSTM_BUF/DynamicEnv_0.9_0.1/success_seed_{seed}.npy')
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
    # label_names.append("TD3_LSTM_BUF_0.9")




    _plot_all(training_timesteps, training_rewards, label_names)
    # _plot_all(training_timesteps, training_success, label_names)

if __name__ == "__main__":
    # TD3:      运行时间: 10650.896821260452   运行时间: 11453.017227172852
    # TD3_LSTM:
    # TD3_LSTM_ATT:  运行时间: 41444.26007390022
    # DDPG:   运行时间: 10509.981415510178
    # SAC:    运行时间: 20815.73937177658    运行时间: 18803.532695770264
    # SAC_LSTM:   运行时间: 54179.132876873016
    # policy_names = ["TD3", "TD3_LSTM", "TD3_LSTM_ATT", "SAC_LSTM", 'SAC']
    # policy_names = ["TD3", "TD3_LSTM", "TD3_LSTM_ATT_2"]
    # env_name = "DynamicEnv"
    # plot_rewards(policy_names, env_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)