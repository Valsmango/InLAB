# coding=utf-8
import torch
from DRLpractice.UAV.UAVsingle.env.envController import *
from DRLpractice.UAV.UAVsingle.algo.TD3 import *
import matplotlib.pyplot as plt
import time

# 设定如果100步内没有到达终点则直接done
MAX_EP_STEPS = 100


# 测试随机（均匀分布！后续可以修改为正态分布）情况下的reward
def random_eval():
    eval_env = getEnv("UAV_single_continuous")
    eval_env.seed(10)
    state, done = eval_env.reset(), False
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action = env.sample_action()
        s, r, done = env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]['delta_v_x'], action[0]['delta_v_y'], action[0]['delta_v_z']}\n"
              f"           State: pos {s[0]['x'], s[0]['y'], s[0]['z']};   speed {s[0]['v_x'], s[0]['v_y'], s[0]['v_z']}\n"
              f"           Reward:{r}\n")
        if done:
            eval_env.show_path()
            break
    eval_env.close()

# 玩1局
def model_eval(policy):
    eval_env = getEnv("UAV_single_continuous")
    eval_env.seed(10)
    state, done = eval_env.reset(), False
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        state = np.array([list(state[i].values()) for i in range(len(state))])
        action = policy.select_action(state)
        s, r, done = env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]['delta_v_x'], action[0]['delta_v_y'], action[0]['delta_v_z']}\n"
              f"           State: pos {s[0]['x'], s[0]['y'], s[0]['z']};   speed {s[0]['v_x'], s[0]['v_y'], s[0]['v_z']}\n"
              f"           Reward:{r}\n")
        state = s
        if done:
            eval_env.show_path()
            break
    eval_env.close()


# 测试
if __name__ == "__main__":
    # 设定环境
    env = getEnv("UAV_single_continuous")
    env.seed(10)
    torch.manual_seed(10)
    np.random.seed(10)

    # 配置model的一些信息
    state_dim = 12
    action_dim = 3
    max_action = np.array([5.0, 5.0, 0.5])
    # 载入训练好的模型
    policy_name = "TD3"
    env_name = "UAV_single_continuous"
    seed_num = 0
    file_name = f"{policy_name}_{env_name}_{seed_num}"
    policy = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                 discount=0.99, tau=0.005, policy_noise=0.2*max_action,
                 noise_clip=0.5*max_action, policy_freq=2)
    policy.load("./model/")
    # 测试随机选择（非正态分布）
    random_eval()
    # 测试模型
    model_eval(policy)

