# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


# 每eval_frequent次训练后会进行一次evaluate（一次evaluate会玩10局），将这些测试所得的rewards画出
def eval_result_plot(rewards, eval_frequent):
    plt.plot(rewards)
    plt.ylabel('average reward')
    plt.xlabel(f'evaluate per {eval_frequent} steps')
    plt.title('Evaluation over 10 episodes')
    plt.show()


def training_rewards_plot(rewards, max_training_steps):
    pass

if __name__ == "__main__":
    policy_name = "TD3"
    env_name = "UAV_single_continuous"
    seed_num = 0
    file_name = f"{policy_name}_{env_name}_{seed_num}"
    rewards = np.load(f"./results/{file_name}.npy")
    eval_result_plot(rewards=rewards, eval_frequent=5e3)