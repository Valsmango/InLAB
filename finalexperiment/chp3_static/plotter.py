# coding=utf-8
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
        pic_timesteps = []
        for i in range(len(rewards)):
            if i == 0:
                ma_rewards.append(rewards[i])
                ma_success.append(success[i])
            else:
                ma_rewards.append(ma_rewards[-1] * 0.9975 + rewards[i] * 0.0025)
                ma_success.append(ma_success[-1] * 0.9975 + success[i] * 0.0025)
            pic_timesteps.append(timesteps[i])
            if timesteps[i] > int(2e6):
                break

        training_timesteps.append(pic_timesteps)
        training_rewards.append(ma_rewards)
        training_success.append(ma_success)
        label_names.append(model_name)


    _plot_all(training_timesteps, training_rewards, label_names)
    # _plot_all(training_timesteps, training_success, label_names)

if __name__ == "__main__":
    # TD3:      运行时间: 17037.15635228157
    # TD3_LSTM:      这里在跑
    # TD3_LSTM_BUF：   运行时间: 55178.274712085724
    # DDPG
    # SAC
    # policy_names = ["TD3", "TD3_LSTM", "TD3_LSTM_ATT", 'SAC']
    # policy_names = ["TD3", "TD3_LSTM", "TD3_LSTM_BUF"]



    policy_names = ["TD3",'TD3_LSTM','SAC','DDPG']
    env_name = "StaticEnv_2e6"
    plot_rewards(policy_names, env_name)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)


    # LSTM_2:把min改成mean
    # ATT: 把actor的注意力删了  ---》 running
    # ATT_2：把归一化
    # ATT_3：输入前加权
    # ATT_4: 把Critic的注意力删了
    #


    # sac_LSTM
    # SAC,把它调低 ---> alpha改成0.3有一丢丢效果， 正在尝试gamma改成0.95
    # ATT4，再试一下加 sqrt dk
    # Buffer搞好之后和ATT结合起来跑跑看
