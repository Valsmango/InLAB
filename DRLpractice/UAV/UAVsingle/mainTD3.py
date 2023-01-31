# coding=utf-8
import gym
import argparse
import os

import numpy as np
import torch

from DRLpractice.UAV.UAVsingle.env import envController
from DRLpractice.UAV.UAVsingle.algo import TD3

import time
from rl_plotter.logger import Logger


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = envController.getEnv(env_name)
    eval_env.seed(seed + 10)
    torch.manual_seed(seed + 10)
    np.random.seed(seed + 10)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            # eval_env.render()
            # time.sleep(0.01)
            state = np.array([list(state[i].values()) for i in range(len(state))])
            action = policy.select_action(state)
            # print(action)
            action = [dict(zip(['delta_v_x', 'delta_v_y', 'delta_v_z'], action))]
            state, reward, done = eval_env.step(action)

            avg_reward += reward
        eval_env.close()
        eval_env = envController.getEnv(args.env)
        # Set seeds
        eval_env.seed(seed + 10)
        torch.manual_seed(seed + 10)
        np.random.seed(seed + 10)
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    starttime = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    # mujoco环境介绍： https://www.jianshu.com/p/e7235f8af25e
    parser.add_argument("--env", default="UAV_single_continuous")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = envController.getEnv(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 12
    action_dim = 3
    max_action = np.array([5.0, 5.0, 0.5])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    # elif args.policy == "OurDDPG":
    #     policy = OurDDPG.OurDDPG(**kwargs)
    # elif args.policy == "OriginDDPG":
    #     policy = OriginDDPG.OriginDDPG(**kwargs)

    if args.load_model != "":
        policy.load(f"./models/{file_name}")

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # env.render()
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample_action()
        else:
            tmp_state = np.array([list(state[i].values()) for i in range(len(state))])
            action = (
                    policy.select_action(tmp_state)
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            action = [dict(zip(['delta_v_x', 'delta_v_y', 'delta_v_z'], action))]

        # Perform action
        next_state, reward, done = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        policy.memory.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            env.close()
            env = envController.getEnv(args.env)
            # Set seeds
            env.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if t >= args.start_timesteps and (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)


    policy.save(f"./models/{file_name}")

    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)
    print(evaluations)

