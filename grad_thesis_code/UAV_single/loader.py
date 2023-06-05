# coding=utf-8
from grad_thesis_code.UAV_single.env.env import Env     # 其实应该用新的环境更好，这里还是用的训练的环境
from grad_thesis_code.UAV_single.DDPG import DDPG
from grad_thesis_code.UAV_single.SAC import SAC
from grad_thesis_code.UAV_single.TD3 import TD3
from grad_thesis_code.UAV_single.TD3_LSTM import TD3LSTM    # TD3_LSTM_BU F只影响训练过程，模型结构和 TD3_LSTM 是一样的
import torch
import matplotlib.pyplot as plt
import time
from tqdm import *
import numpy as np

# 设定如果100步内没有到达终点则直接done
MAX_EP_STEPS = 200

def run_model(env_name, policy_name):
    batch_size = 64
    state_dim = 12
    action_dim = 3
    max_action = 1.0

    if policy_name == "DDPG":
        agent = DDPG(state_dim, action_dim, max_action)
    elif policy_name == "TD3":
        agent = TD3(state_dim, action_dim, max_action)
    elif policy_name == "SAC":
        agent = SAC(state_dim, action_dim, max_action)
    elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
        agent = TD3LSTM(batch_size, state_dim, action_dim, max_action)
    else:
        agent = None

    agent.load(f"./model_train/{policy_name}/{env_name}/{policy_name}")

    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.
    eval_env = Env()
    s, done = eval_env.reset(), False
    reward = 0.0
    if policy_name == "TD3_LSTM" or policy_name == "SAC_LSTM" \
            or policy_name == "TD3_LSTM_ATT" or policy_name == "TD3_LSTM_BUF":
        h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
    for j in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        if policy_name == "TD3" or policy_name == "DDPG":
            a = agent.choose_action(s)
        elif policy_name == "SAC":
            a = agent.choose_action(s, deterministic=True)
        elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_ATT" \
                or policy_name == "TD3_LSTM_BUF":
            a, h, c = agent.choose_action(s, h, c)

        s_, r, done, category = eval_env.step(a)
        s = s_
        reward += r
        if done:
            cur_path_len = eval_env.get_path_len()
            cur_path_delay = eval_env.get_path_delay()
            path_len += cur_path_len
            path_delay += cur_path_delay
            if category == 1:
                success_path_len += cur_path_len
                success_path_delay += cur_path_delay
            break

    eval_env.show_path()
    eval_env.close()
    return reward, path_len, path_delay, success_path_len, success_path_delay


def test_model(env_name, policy_name):
    batch_size = 64
    state_dim = 12
    action_dim = 3
    max_action = 1.0

    if policy_name == "DDPG":
        agent = DDPG(state_dim, action_dim, max_action)
    elif policy_name == "TD3":
        agent = TD3(state_dim, action_dim, max_action)
    elif policy_name == "SAC":
        agent = SAC(state_dim, action_dim, max_action)
    elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
        agent = TD3LSTM(batch_size, state_dim, action_dim, max_action)
    else:
        agent = None

    agent.load(f"./model_train/{policy_name}/{env_name}/{policy_name}")

    success_times = 0
    collision_times = 0
    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.
    for i in tqdm(range(1000)):
        eval_env = Env()
        s, done = eval_env.reset(), False
        reward = 0.0
        if policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
            h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
        for j in range(MAX_EP_STEPS):
            if policy_name == "TD3" or policy_name == "DDPG":
                a = agent.choose_action(s)
            elif policy_name == "SAC":
                a = agent.choose_action(s, deterministic=True)
            elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
                a, h, c = agent.choose_action(s, h, c)

            s_, r, done, category = eval_env.step(a)
            s = s_
            reward += r
            if done:
                cur_path_len = eval_env.get_path_len()
                cur_path_delay = eval_env.get_path_delay()
                path_len += cur_path_len
                path_delay += cur_path_delay
                if category == 1:
                    success_times += 1
                    success_path_len += cur_path_len
                    success_path_delay += cur_path_delay
                elif category == 2:
                    collision_times += 1
                break
    if success_times == 0:
        return success_times, collision_times, path_len / 1000, path_delay / 1000, 0, 0
    else:
        return success_times, collision_times, path_len / 1000, path_delay / 1000, success_path_len / success_times, success_path_delay / success_times


if __name__ == '__main__':
    # 注意要先固定起终点，即修改env

    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env_name = "DynamicEnv"     # DynamicEnv、StaticEnv
    policy_name = "TD3"     # DDPG、TD3、TD3_LSTM、TD3_LSTM_BUF、SAC

    # # 表格数据：统计多次的测试结果均值
    # success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model(env_name, policy_name)
    # print(f"success times:{success}    collision times:{collision}      "
    #       f"path length:{path_len}      path delay:{path_delay}       "
    #       f"success path length:{success_path_len}      success path delay:{success_path_delay}")

    # 用于画图：单次测试对应的单条路径信息
    reward, path_len, path_delay, success_path_len, success_path_delay = run_model(env_name, policy_name)
    print(f"reward:{reward}    path length:{path_len}      path delay:{path_delay}       "
          f"success path length:{success_path_len}      success path delay:{success_path_delay}")



