# coding=utf-8
import os
import random
import time
from tqdm import tqdm

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter

from DRLpractice.MujocoAlgorithms.mymodels import PPO

import time

if __name__ == "__main__":
    starttime = time.time()
    eval_records = []

    # Env parameters
    model_name = "PPO"
    env_name = "Hopper-v2"
    seed = 10
    # writer = SummaryWriter("./runs/tensorboard/ppo")

    # Set gym environment
    env = gym.make(env_name)
    # state_dim = 12
    # action_dim = 3
    # # n_actions = env.action_space.n
    # # try:
    # #     n_states = env.observation_space.n
    # # except AttributeError:
    # #     n_states = env.observation_space.shape[0]
    # max_action = np.array([1.0, 1.0, 1.0])
    state_dim = env.observation_space.shape[0]  # 对于Hopper，state_dim = 11
    action_dim = env.action_space.shape[0]  # 对于Hopper，action_dim = 3
    max_action = float(env.action_space.high[0])  # 对于Hopper，max_action = 1.0，并且，env.action_space.high=[-1. -1. -1.]

    # Set seeds
    env.seed(seed)
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
    max_episodes = 200
    clip = 0.2
    gamma = 0.99
    gae_lambda = 0.92
    step_count = 0
    epochs = 10
    update_frequent = 20

    # Initiate the network and set the optimizer
    model = PPO.PPO(batch_size=batch_size, state_space=state_dim, action_space=action_dim, clip=clip,
                    actor_lr=actor_learning_rate, critic_lr=critic_learning_rate,
                    epochs=epochs, gamma=gamma, gae_lambda=gae_lambda)

    # output the reward
    rewards = []
    ma_rewards = []
    print_per_iter = 20  # 每玩1把游戏进行一次结果输出
    score = 0
    score_sum = 0.0

    # 开始训练
    for i in range(max_episodes):
        # Initialize the environment and state
        state, done = env.reset(), False
        while not done:
            # 渲染
            env.render()
            # Select and perform an action
            action, prob, val = model.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
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