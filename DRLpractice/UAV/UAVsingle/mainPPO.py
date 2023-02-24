# coding=utf-8
import os
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter
from DRLpractice.UAV.UAVsingle.env import envController
from DRLpractice.UAV.UAVsingle.algo import PPO

import time

'''
https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

下面这个很完善！！！！
https://zhuanlan.zhihu.com/p/512327050
https://github.com/Lizhi-sjtu/DRL-code-pytorch

'''


if __name__ == "__main__":
    starttime = time.time()
    eval_records = []

    # Env parameters
    model_name = "PPO"
    env_name = "UAV_single_continuous"
    seed = 10
    # writer = SummaryWriter("./runs/tensorboard/ppo")

    # Set gym environment
    env = envController.getEnv(env_name)
    state_dim = 12
    action_dim = 3
    max_action = np.array([1.0, 1.0, 1.0])

    # Set seeds
    # env.action_space.seed(seed)     # 这个是TD3中提到的seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # save results
    file_name = f"{model_name}_{env_name}_{seed}"
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    print("---------------------------------------")
    print(f"Policy: {model_name}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")


    # model parameters
    # 适当设置batch size，过小的经验池容量和batchsize导致收敛到局部最优，结果呈现震荡形式
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-3
    batch_size = 16
    max_episodes = int(5e4)
    clip = 0.2
    gamma = 0.99
    gae_lambda = 0.92
    step_count = 0
    epochs = 10
    update_frequent = 20    # 每走20步更新一次，太多的话可能当前的episode都结束了

    # Initiate the network and set the optimizer
    model = PPO.PPO(batch_size=batch_size, state_space=state_dim, action_space=action_dim, clip=clip,
                        actor_lr=actor_learning_rate, critic_lr=critic_learning_rate,
                        epochs=epochs, gamma=gamma, gae_lambda=gae_lambda, max_action=max_action)

    # output the reward
    rewards = []
    ma_rewards = []
    print_per_iter = 200     # 每玩200把游戏进行一次结果输出
    score = 0
    score_sum = 0.0

    # 开始训练
    for i in tqdm(range(max_episodes)):
        # Initialize the environment and state
        env = envController.getEnv(env_name)
        env.seed(seed)
        state = env.reset()
        done = None
        while not done:
            # 渲染
            # env.render()
            # Select and perform an action
            tmp_state = np.array([list(state[i].values()) for i in range(len(state))])
            action, prob, val = model.select_action(tmp_state)
            # action = (
            #         action + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            # ).clip(-max_action, max_action)
            # print(action)
            action = np.clip(action, -1, 1)
            action = [dict(zip(['delta_v_x', 'delta_v_y', 'delta_v_z'], action))]
            # print(action)
            next_state, reward, done = env.step(action)
            # next_state = next_state[::2]
            # reward是一个float格式的数值
            score += reward
            score_sum += reward
            done_bool = float(done)
            model.memory.add(s=state, a=action, r=reward, done=done_bool, p=prob, v=val)

            # Move to the next state
            state = next_state
            step_count += 1

            # Perform one step of the optimization
            if step_count % update_frequent == 0:
                model.update()
        env.close()
        rewards.append(score)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * score)
        else:
            ma_rewards.append(score)
        # 隔一段时间，输出一次训练的结果
        if i % print_per_iter == 0 and i != 0:
            print("n_episode :{}, score : {:.1f}".format(i, score))
        # writer.add_scalar("rewards", score, i + 1)
        score = 0
    # save the model
    model.save(f"./models/{file_name}")

    endtime = time.time()
    dtime = endtime - starttime
    print("-----------------------------------------")
    print("程序运行时间：%.8s s" % dtime)
    print("-----------------------------------------")

    plt.plot(rewards)
    plt.ylabel('reward')
    plt.xlabel(f'episode')
    plt.title('Training Reward')
    plt.show()

    plt.plot(ma_rewards)
    plt.ylabel('reward')
    plt.xlabel(f'episode')
    plt.title('Training MA Reward')
    plt.show()