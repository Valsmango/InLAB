# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

'''
相关的一些博客：
http://www.deeprlhub.com/d/114/8
https://github.com/gxywy/rl-plotter

https://www.zhihu.com/question/559255026
https://zhuanlan.zhihu.com/p/75477750
https://mp.weixin.qq.com/s?__biz=MzU1NjEwMTY0Mw==&mid=2247543585&idx=1&sn=14a6b883e174275d511559e0157291bc&chksm=fbc85245ccbfdb53a0f7119d3c679fca6f499f833d58a0b2e9c512fdff777805fe552bddaa7b&scene=27

https://www.bilibili.com/video/BV1st411z7ZC?p=12

'''
# 红色 178/255, 34/255, 34/255   205/255, 51/255, 51/255
# 橙色 244/255, 164/255, 96/255,
# 绿色 46/255, 139/255, 87/255
# 蓝色 70/255, 130/255, 180/255
# 紫色 106/255, 90/255, 205/255
color_list = [(205/255, 51/255, 51/255), (244/255, 164/255, 96/255), (46/255, 139/255, 87/255), (106/255, 90/255, 205/255), (70/255, 130/255, 180/255)]

def all_eval_result_plot(rewards, eval_frequent):
    # 输入rewards为：eval episodes（eval总次数） ， 40（每一次eval的重复次数） --> transpose后为：40 x episodes
    rewards = np.transpose(rewards)
    df = pd.DataFrame(rewards).melt(var_name="eval_times", value_name="rewards")
    sns.lineplot(x="eval_times", y="rewards", data=df)
    plt.ylabel('average reward')
    plt.xlabel(f'evaluate per {eval_frequent} steps')
    plt.title('Evaluation over 40 episodes')
    plt.show()

def test_plot(x, y, policy_name):
    sns.set_style('whitegrid')
    plt.plot(x, y, label=policy_name)
    plt.show()

def all_training_ma_plot(rewards, policy_names, xlabel, ylabel, title):
    sns.set_style('whitegrid')
    for reward, name in zip(rewards, policy_names):
        plt.plot(reward, label=name)    # alpha = 0.5 透明度
        # plt.plot(reward[0:20000], label=name)  # alpha = 0.5 透明度
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.show()


def fig_rewards(one_hot, env_name="StandardEnv"):
    training_ma_rewards = []
    label_names = []
    seed_num = 10

    if one_hot[0]:
        policy_name = "DDPG"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("DDPG")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))

    if one_hot[1]:
        policy_name = "DDPGLSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-DDPG")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))

    if one_hot[2]:
        policy_name = "TD3"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("TD3")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))

    if one_hot[3]:
        policy_name = "TD3LSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-TD3")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))

    if one_hot[4]:
        policy_name = "SAC"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("SAC")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))

    if one_hot[5]:
        policy_name = "SACLSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-SAC")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))

    if one_hot[6]:
        policy_name = "TD3LSTMATT"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("RA-TD3")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))

    if one_hot[7]:
        policy_name = "TD3LSTM_buffer"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("TD3LSTM_buffer")
        training_ma_rewards.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_reward_{file_name}.npy"))



    if env_name == "StandardEnv":
        title_name = "Scence 1"
    else:
        title_name = "Scence 2"
    fig_title = f"Training - {title_name}"
    # fig_title = "Training"
    all_training_ma_plot(training_ma_rewards, label_names, "Episode", "Episode Reward", fig_title)


def fig_success_rate(one_hot, env_name="StandardEnv"):
    training_ma_success = []
    label_names = []
    seed_num = 10

    if one_hot[0]:
        policy_name = "DDPG"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("DDPG")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))

    if one_hot[1]:
        policy_name = "DDPGLSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-DDPG")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))

    if one_hot[2]:
        policy_name = "TD3"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("TD3")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))

    if one_hot[3]:
        policy_name = "TD3LSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-TD3")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))

    if one_hot[4]:
        policy_name = "SAC"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("SAC")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))

    if one_hot[5]:
        policy_name = "SACLSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-SAC")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))

    if one_hot[6]:
        policy_name = "TD3LSTMATT"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("RA-TD3")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))

    if one_hot[7]:
        policy_name = "TD3LSTM_buffer"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("TD3LSTM_buffer")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_success_{file_name}.npy"))


    if env_name == "StandardEnv":
        title_name = "Scence 1"
    else:
        title_name = "Scence 2"
    # fig_title = f"Training - {title_name}"
    fig_title = "Training"
    all_training_ma_plot(training_ma_success, label_names, "Episode", "Success Rate", fig_title)


def fig_collision_rate(one_hot, env_name="StandardEnv"):
    training_ma_success = []
    label_names = []
    seed_num = 10

    if one_hot[0]:
        policy_name = "DDPG"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("DDPG")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))

    if one_hot[1]:
        policy_name = "DDPGLSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-DDPG")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))

    if one_hot[2]:
        policy_name = "TD3"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("TD3")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))

    if one_hot[3]:
        policy_name = "TD3LSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-TD3")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))

    if one_hot[4]:
        policy_name = "SAC"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("SAC")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))

    if one_hot[5]:
        policy_name = "SACLSTM"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("LSTM-SAC")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))

    if one_hot[6]:
        policy_name = "TD3LSTMATT"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("RA-TD3")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))

    if one_hot[7]:
        policy_name = "TD3LSTM_buffer"
        file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
        label_names.append("TD3LSTM_buffer")
        training_ma_success.append(np.load(f"./eval_reward_train/{policy_name}/train_ma_collision_{file_name}.npy"))


    if env_name == "StandardEnv":
        title_name = "Scence 1"
    else:
        title_name = "Scence 2"
    # fig_title = f"Training - {title_name}"
    fig_title = "Training"
    all_training_ma_plot(training_ma_success, label_names, "Episode", "Collision Rate", fig_title)


if __name__ == "__main__":
    ###################################### Rewards ###################################
    # 对应 DDPG, DDPG-LSTM, TD3, TD3-LSTM, SAC, SAC-LSTM, TD3-LSTM-ATT, TD3-LSTM-BUFFER
    # one_hot = [True, False, True, True, True, True, False]
    # fig_rewards(one_hot, "StandardEnv")
    one_hot = [False, False, True, True, False, False, True, True]
    fig_rewards(one_hot, "DynamicEnv")

    ################################## Success Rate ##################################
    # 对应 DDPG, DDPG-LSTM, TD3, TD3-LSTM, SAC, SAC-LSTM, TD3-LSTM-ATT
    # one_hot = [False, False, False, True, False, True, False]
    # fig_success_rate(one_hot, "StandardEnv")
    # one_hot = [False, False, True, False, True, False, True]
    # fig_success_rate(one_hot, "DynamicEnv")

    ################################## Collision Rate ##################################
    # 对应 DDPG, DDPG-LSTM, TD3, TD3-LSTM, SAC, SAC-LSTM, TD3-LSTM-ATT
    # one_hot = [False, False, False, True, False, True, False]
    # fig_collision_rate(one_hot, "StandardEnv")
    # one_hot = [False, False, True, False, True, False, True]
    # fig_collision_rate(one_hot, "DynamicEnv")

    # policy_name = "TD3LSTM"
    # env_name ="DynamicEnv"
    # seed_num = 10
    # file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
    # rewards = np.load(f"./eval_reward_train/{policy_name}/train_reward_{file_name}.npy")
    # ma_rewards = []
    # x = np.arange(0, len(rewards))*2
    # for i in range(len(rewards)):
    #     if i == 0:
    #         ma_rewards.append(rewards[i])
    #     else:
    #         ma_rewards.append(ma_rewards[-1] * 0.99 + rewards[i]*0.01)
    #
    # test_plot(x, ma_rewards, policy_name)