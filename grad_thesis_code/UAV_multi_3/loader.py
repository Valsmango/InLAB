# coding=utf-8
from grad_thesis_code.UAV_multi_3.env.env import Env
from grad_thesis_code.UAV_multi_3.TD3 import TD3
from grad_thesis_code.UAV_multi_3.TD3_LSTM import TD3LSTM
import time
import numpy as np
import torch


if __name__ == "__main__":
    state_dim = 15
    action_dim = 3
    max_action = 1.0
    n_agents = 3
    # 多机场景中仅进行了单次路径展示
    # 注意要先固定起终点，即修改env

    ##################### load TD3_LSTM_BUF ######################
    env_name = "MAEnv"
    policy_name = "TD3"

    env = Env()

    # Set random seed
    seed = 20
    env.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    max_episode_steps = 200  # Maximum number of steps per episode
    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration

    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.

    if policy_name == "TD3":
        agent = [TD3(state_dim, action_dim, max_action) for _ in range(n_agents)]
    elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
        agent = [TD3LSTM(1, state_dim, action_dim, max_action) for _ in range(n_agents)]
    else:
        agent = None

    for i in range(n_agents):
        # agent[i].load(f"./model_train/{policy_name}/{env_name}/agent_{i}_{policy_name}")
        # 造假的图更好看……但其实该换成TD3_LSTM_BUF的，论文里面偷懒了
        agent[i].load(f"./model_train/{policy_name}_fake/{env_name}/agent_{i}_{policy_name}_MAStandardEnv")

    episode_reward = 0.
    s, done = env.reset(), [False for i in range(n_agents)]
    h = []
    c = []

    if policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
        for i in range(n_agents):
            h_i, c_i = agent[i].actor.init_hidden_state(batch_size=1, training=False)
            h.append(h_i)
            c.append(c_i)

    for j in range(max_episode_steps):
        env.render()
        time.sleep(0.1)
        if policy_name == "TD3":
            a = [agent[i].choose_action(s[i]) for i in range(n_agents)]
        elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
            a = []
            for i in range(n_agents):
                a_i, h[i], c[i] = agent[i].choose_action(s[i], h[i], c[i])
                # a_i = (a_i + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                a.append(a_i)
        else:
            a = None
        s_, r, done, store_flags = env.step(a)
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