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


# 每eval_frequent次训练后会进行一次evaluate（一次evaluate会玩10局），将这些测试所得的rewards画出
def avg_eval_result_plot(rewards, eval_frequent):
    plt.plot(rewards)
    plt.ylabel('average reward')
    plt.xlabel(f'evaluate per {eval_frequent} steps')
    plt.title('Evaluation over 40 episodes')
    plt.show()


def all_eval_result_plot(rewards, eval_frequent):
    rewards = np.transpose(rewards)
    df = pd.DataFrame(rewards).melt(var_name="eval_times", value_name="rewards")
    sns.lineplot(x="eval_times", y="rewards", data=df)
    plt.ylabel('average reward')
    plt.xlabel(f'evaluate per {eval_frequent} steps')
    plt.title('Evaluation over 40 episodes')
    plt.show()


def training_rewards_plot(rewards, max_training_steps):
    pass


if __name__ == "__main__":
    # ####################################  载入PPO  ##########################################
    # policy_name = "PPO_Beta"
    # env_name = "StandardEnv"
    # seed_num = 10
    # file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
    # avg_rewards = np.load(f"./eval_reward_train/{file_name}.npy")
    # avg_eval_result_plot(rewards=avg_rewards, eval_frequent=5000)
    # # # all_rewards = np.load(f"./eval_reward_train/{file_name}.npy")
    # # # all_eval_result_plot(rewards=all_rewards, eval_frequent=5000)

    # ####################################  载入TD3  ##########################################
    # policy_name = "TD3"
    # env_name = "StandardEnv"
    # seed_num = 10
    # file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
    # avg_rewards = np.load(f"./eval_reward_train/{file_name}.npy")
    # avg_eval_result_plot(rewards=avg_rewards, eval_frequent=5000)

    # ####################################  载入SAC  ##########################################
    # policy_name = "SAC"
    # env_name = "StandardEnv"
    # seed_num = 10
    # file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
    # avg_rewards = np.load(f"./eval_reward_train/{file_name}.npy")
    # avg_eval_result_plot(rewards=avg_rewards, eval_frequent=5000)

    ####################################  载入DDPG  ##########################################
    policy_name = "DDPG"
    env_name = "StandardEnv"
    seed_num = 10
    file_name = f"{policy_name}_env_{env_name}_seed_{seed_num}"
    avg_rewards = np.load(f"./eval_reward_train/{file_name}.npy")
    avg_eval_result_plot(rewards=avg_rewards, eval_frequent=5000)