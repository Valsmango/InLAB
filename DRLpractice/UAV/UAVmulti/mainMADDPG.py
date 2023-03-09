# coding=utf-8
"""
main：
    每一个episode：
        obs = env.reset()
        agent_actions = maddpg.step() # step也就是choose_action，一个maddpg中包括多个ddpg的agent，每个ddpg的agent去执行choose_action
        从里面提取出每个agent的action，然后进行格式转换（rearrange） --> actions
        next_obs, rewards, dones, infos = env.step(actions)
        replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
        obs = next_obs

"""
import gym
import argparse
import os

import numpy as np
import torch

from DRLpractice.UAV.UAVmulti.Env import Env
from DRLpractice.UAV.UAVmulti.MADDPG import MADDPG, MADDPGReplayBuffer

import time


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

def eval_policy(maddpg):
    times = 40  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    n_agents = 2
    # seed = 20
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    for _ in range(times):
        env = Env(mode="test")
        observations = env.reset()
        dones = [False for i in range(n_agents)]
        episode_reward = 0
        while not np.any(np.array(dones)):
            actions = maddpg.choose_actions(observations=observations)
            next_observations, rewards, dones = env.step(actions)
            episode_reward += sum(rewards)
            observations = next_observations
        env.close()
        evaluate_reward += episode_reward

    return evaluate_reward / times


if __name__ == "__main__":

    starttime = time.time()
    eval_records = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    seed = 10
    env_name = "MAStandardEnv"
    env = Env(mode="train")
    policy_name = "MADDPG"
    state_dim = 12
    action_dim = 3
    max_action = 1.0
    n_agents = 2
    max_timesteps = 2e6
    start_timesteps = 25e3
    eval_freq = 5e3
    expl_noise = 0.1  # Std of Gaussian exploration noise
    batch_size = 64

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    file_name = f"{policy_name}_env_{env_name}_seed_{seed}"
    print("---------------------------------------")
    print(f"Policy: {policy_name}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    if not os.path.exists("./eval_reward_train/MADDPG/"):
        os.makedirs("./eval_reward_train/MADDPG/")

    if not os.path.exists("./model_train/MADDPG/"):
        os.makedirs("./model_train/MADDPG/")

    maddpg = MADDPG(n_agents)
    replay_buffer = MADDPGReplayBuffer(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_size=int(max_timesteps))
    # explr_pct_remaining = max(0, n_exploration_eps - ep_i) / config.n_exploration_eps
    # maddpg.scale_noise(
    #     config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
    # maddpg.reset_noise()

    # Evaluate untrained policy
    evaluate_rewards = []
    evaluate_num = 0
    evaluate_freq = 5e3

    observations, dones = env.reset(), [False for i in range(n_agents)]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # env.render()
        # 初始不再是xxx步用于随机探索（即Select action randomly or according to policy）
        # 全都是根据policy
        if t < start_timesteps:
            actions = env.sample_action()
        else:
            actions = maddpg.choose_actions(observations=observations)

        # Perform action
        next_observations, rewards, dones = env.step(actions)
        done_bools = [float(dones[i]) if episode_timesteps < env._max_episode_steps else 0 for i in range(n_agents)]

        # Store data in replay buffer
        replay_buffer.add(observations=observations, actions=actions, rewards=rewards, next_observations=next_observations, dones=done_bools)
        observations = next_observations
        episode_reward += sum(rewards)

        # Train agent after collecting sufficient data
        # 只训练了一次，DDPG那个代码里面是隔50步训练50次……但标准的TD3其实也是走一步训练一次
        if t >= start_timesteps:
            for agent_i in range(n_agents):
                sample = replay_buffer.sample(batch_size)
                maddpg.update(sample, agent_i)
            maddpg.update_all_targets()

        # if t >= start_timesteps and (t + 1) % evaluate_freq == 0:
        if (t + 1) % evaluate_freq == 0:
            evaluate_num += 1
            evaluate_reward = eval_policy(maddpg)
            evaluate_rewards.append(evaluate_reward)
            # print("---------------------------------------")
            print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
            # print("---------------------------------------")
            # writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
            # Save the rewards
            if evaluate_num % 10 == 0:
                np.save('./eval_reward_train/MADDPG/MADDPG_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(evaluate_rewards))

        if np.any(np.array(dones)):
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            env.close()
            env = Env(mode="train")
            observations, done = env.reset(), [False for i in range(n_agents)]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    env.close()
    maddpg.save(f"./model_train/MADDPG/MADDPG_{env_name}")

    endtime = time.time()
    dtime = endtime - starttime
    end_time = time.time()
    print("程序运行时间：%.8s s" % dtime)
