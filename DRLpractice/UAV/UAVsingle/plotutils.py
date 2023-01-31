# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

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
    eval_result_plot(rewards=rewards, eval_frequent=5000)