# coding=utf-8
import time
from DRLpractice.UAV.UAVmulti.Env import Env
from DRLpractice.UAV.UAVmulti.MADDPG import MADDPG
import numpy as np

MAX_EP_STEPS = 200


def model_eval_MADDPG(maddpg, n_agents):
    eval_env = Env("test")
    eval_env.seed(20)
    observations = eval_env.reset()
    dones = [False for i in range(n_agents)]
    total_reward = 0.0
    i = 0
    while not np.any(np.array(dones)):
        eval_env.render()
        time.sleep(0.1)
        actions = maddpg.choose_actions(observations=observations)
        next_observations, rewards, dones = eval_env.step(actions)
        print(f"currently, the {i + 1} step:\n"
              f"    Action: speed {actions[0][0] * 5.0, actions[0][1] * 5.0, actions[0][2] * 0.5}\n"
              f"    State: pos {observations[0][0] * 5000.0, observations[0][1] * 5000.0, observations[0][2] * 300.0}\n"
              f"           speed {observations[0][3] * 200, observations[0][4] * 200.0, observations[0][5] * 10}\n "
              f"    Reward:{rewards[0], rewards[1]}\n")
        total_reward += sum(rewards)
        observations = next_observations
        i += 1
    print(total_reward)
    eval_env.show_path()
    eval_env.close()


if __name__ == "__main__":
    state_dim = 12
    action_dim = 3
    max_action = 1.0
    n_agents = 2
    max_timesteps = 1e6
    policy_name = "MADDPG"
    env_name = "MAStandardEnv"
    file_name = f"./model_train/MADDPG/{policy_name}_{env_name}"
    policy = MADDPG(n_agents)
    policy.load(file_name)
    # 测试随机选择（非正态分布）
    # random_eval()
    # 测试模型
    model_eval_MADDPG(policy, n_agents)
