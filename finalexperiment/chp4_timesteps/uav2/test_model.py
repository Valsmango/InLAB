# coding=utf-8
from finalexperiment.chp4_timesteps.uav2.env.eval_env import Env
from finalexperiment.chp4_timesteps.uav2.TD3 import TD3
from finalexperiment.chp4_timesteps.uav2.TD3_LSTM_BUF import TD3LSTM
import torch
import matplotlib.pyplot as plt
import time
from tqdm import *
import numpy as np

MAX_EP_STEPS = 200


def model_eval_DDPG(ddpg, n_agents):
    # 适用于DDPG、TD3
    eval_env = Env()
    eval_env.seed(20)
    observations = eval_env.reset()
    dones = [False for _ in range(n_agents)]
    total_reward = 0.0
    i = 0
    while not np.all(np.array(dones)):
        eval_env.render()
        time.sleep(0.1)
        actions = [ddpg[i].choose_action(observations[i]) for i in range(n_agents)]
        next_observations, rewards, dones, _ = eval_env.step(actions)
        print(f"currently, the {i + 1} step:\n"
              f"    Action: speed {actions[0][0] * 5.0, actions[0][1] * 5.0, actions[0][2] * 0.5}\n"
              f"    State: pos {observations[0][0] * 5000.0, observations[0][1] * 5000.0, observations[0][2] * 300.0}\n"
              f"           speed {observations[0][3] * 200, observations[0][4] * 200.0, observations[0][5] * 10}\n "
              f"    Reward:{rewards[0]}\n"
              f"    Action: speed {actions[1][0] * 5.0, actions[1][1] * 5.0, actions[1][2] * 0.5}\n"
              f"    State: pos {observations[1][0] * 5000.0, observations[1][1] * 5000.0, observations[1][2] * 300.0}\n"
              f"           speed {observations[1][3] * 200, observations[1][4] * 200.0, observations[1][5] * 10}\n "
              f"    Reward:{rewards[1]}\n")
        total_reward += sum(rewards)
        observations = next_observations
        i += 1
    print(total_reward)
    print(eval_env.success_count)
    eval_env.show_path()
    eval_env.close()


if __name__ == '__main__':
    # 多机场景中仅进行了单次路径展示

    ##################### load TD3_LSTM_BUF ######################
    env_name ="MAEnv"
    policy_name = "TD3_LSTM_BUF"

    # Set random seed
    seed = 10
    # env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    state_dim = 12
    action_dim = 3
    max_action = 1.0
    n_agents = 2
    max_episode_steps = 200  # Maximum number of steps per episode
    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration

    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.

    if policy_name == "TD3":
        agent = [TD3(state_dim, action_dim, max_action) for _ in range(n_agents)]
    elif policy_name == "TD3_LSTM"or policy_name=="TD3_LSTM_BUF":
        agent = [TD3LSTM(1, state_dim, action_dim, max_action) for _ in range(n_agents)]

    for i in range(n_agents):
        agent[i].load(f"./model_train/{policy_name}/{env_name}/agent_{i}_{policy_name}")

    episode_reward = 0.
    env = Env()
    s, done = env.reset(), [False for i in range(n_agents)]
    h = []
    c = []

    if policy_name == "TD3_LSTM"or policy_name=="TD3_LSTM_BUF":
        for i in range(n_agents):
            h_i, c_i = agent[i].actor.init_hidden_state(batch_size=1, training=False)
            h.append(h_i)
            c.append(c_i)

    for j in range(max_episode_steps):
        env.render()
        time.sleep(0.1)
        if policy_name == "TD3":
            a = [agent[i].choose_action(s[i]) for i in range(n_agents)]
        elif policy_name == "TD3_LSTM"or policy_name=="TD3_LSTM_BUF":
            a = []
            for i in range(n_agents):
                a_i, h[i], c[i]= agent[i].choose_action(s[i], h[i], c[i])
                # a_i = (a_i + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                a.append(a_i)
        s_, r, done,store_flags = env.step(a)
        s = s_
        episode_reward += sum(r)
        if np.all(np.array(done)):
            cur_path_len = env.get_path_len()
            cur_path_delay = env.get_path_delay()
            path_len += cur_path_len
            path_delay += cur_path_delay
            if env.success_count == n_agents:
                success_path_len += cur_path_len
                success_path_delay += cur_path_delay
            break
    env.close()
    env.show_path()


