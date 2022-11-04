import gym
import argparse
import os

import numpy as np
import torch


import time

# TD3源码：https://github.com/sfujim/TD3
# AE-DDPG源码：https://github.com/anita-hu/TF2-RL/blob/master/AE-DDPG/TF2_AE_DDPG.py


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
from DRLpractice.MujocoAlgorithms import myutils
from DRLpractice.MujocoAlgorithms.mymodels import DDPG_RW


def eval_policy(policy, env_name, seed, eval_episodes):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)   # env的seed固定为100（因为是测试，所以不为0）

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward





if __name__ == "__main__":
    starttime = time.time()

    """
    初始化参数、环境，根据console输入来运行对应的模型
    
    把所有的参数放进arg里面封装起来:
    policy: 默认为DDPG_RW
    env: 默认为Hopper-v2
    seed: TD3中默认为0
    eval_freq: TD3中默认每 5e3 步进行一次eval测试
    max_timesteps：TD3中默认最大步长为 1e6 步
    batch_size：TD3中默认为256，也就是每256个transition进行一次更新；  RW那篇用的是96，而TD3中用的是256
    discount：折现因子，TD3中默认为0.99
    tau：参数更新中的？？？？TD3中默认为0.005;   RW那篇用的是0.125，而TD3中用的是5e-3
    policy_freq：fixed-target网络的更新频率，TD3中默认为2
    save_model：
    load_model：
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG_RW")  # Policy name (TD3, DDPG, OurDDPG & DDPG_RW)
    parser.add_argument("--env", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    # 增加一个参数 eval_episodes
    parser.add_argument("--eval_episodes", default=10, type=int)

    # parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment

    # parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.01)  # Std of Gaussian exploration noise

    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    # parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    # parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    """
    设置默认的环境存储文件名，以及创建存储的文件夹/打开模型所在的文件夹
    存储：
        训练好的模型的存储，放在mymodels中
        结果数据，放在myresults中
    """
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./myresults"):
        os.makedirs("./myresults")

    if args.save_model and not os.path.exists("./mymodels"):
        os.makedirs("./mymodels")

    """
    根据参数make环境,进而获取环境参数、载入对应模型（所以这里的env是训练的env，而测试是在eval——env环境下测试）
    固定随机种子
    """
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]  # 对于Hopper，state_dim = 11
    action_dim = env.action_space.shape[0]  # 对于Hopper，action_dim = 3
    max_action = float(env.action_space.high[0])  # 对于Hopper，max_action = 1.0，并且，env.action_space.high=[-1. -1. -1.]

    # 模型需要的参数，用字典形式包装起来
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "DDPG_RW":
        # Target policy smoothing is scaled wrt the action scale
        # kwargs["policy_noise"] = args.policy_noise * max_action
        # kwargs["noise_clip"] = args.noise_clip * max_action
        # kwargs["policy_freq"] = args.policy_freq
        # policy = TD3.TD3(**kwargs)
        policy = DDPG_RW.DDPG_RW(**kwargs)
    elif args.policy == "DDPG":
        # policy = DDPG.DDPG(**kwargs)
        pass

    # 当前期save过训练好的model，之后就可以用load_model调出来
    # 那么，如果调出来了，前面定义policy有什么用呢？？？
    if args.load_model != "":
        # 如果该输入参数是default，就直接为file_name（默认的存储文件名）；否则就用用户输入的load_model值
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./mymodels/{policy_file}")

    # 初始化replay_buffer
    replay_buffer = myutils.ReplayBuffer(state_dim, action_dim)
    # ！！！改进！！！
    high_replay_buffer = myutils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    # 采用初始的policy来玩游戏，玩eval_episodes=10个episodes，得到的reward均值（也就是evaluations数组的第一个元素为初始情况reward）
    evaluations = [eval_policy(policy, args.env, args.seed, args.eval_episodes)]

    """
    开始训练/update网络参数
    """
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0


    # 生成一个 timesteps长的RW序列
    # 每次选择action就用对应的作为noise
    rw = np.random.normal(0, max_action * args.expl_noise, size=(int(args.max_timesteps), action_dim))
    rw = np.cumsum(rw, axis=0)

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            # action = (
            #         policy.select_action(np.array(state))
            #         + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            # ).clip(-max_action, max_action)
            action = (
                    policy.select_action(np.array(state))
                    + rw[t]*0.05
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0    # 超过_max_episode_steps是0？？？ 也就是done_bool为false？？？

        # # ！！！改进！！！
        # if reward > replay_buffer.reward.max():
        #     high_replay_buffer.add(state, action, next_state, reward, done_bool)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, high_replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

            # 改进！！！ 在一个epi结束后来比较reward,将reward高的episode中每一步存入high_replay_buffer,除非改replay buffer的代码,不然很难做
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed, args.eval_episodes))
            np.save(f"./myresults/{file_name}", evaluations)
            if args.save_model: policy.save(f"./mymodels/{file_name}")

    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)



