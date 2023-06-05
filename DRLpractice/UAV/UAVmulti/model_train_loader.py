# coding=utf-8
from DRLpractice.UAV.UAVmulti.MATD3 import MATD3
from DRLpractice.UAV.UAVmulti.MADDPG import MADDPG
from DRLpractice.UAV.UAVmulti.MASAC import MASAC
from DRLpractice.UAV.UAVmulti.TD3 import TD3
from DRLpractice.UAV.UAVmulti.SAC import SAC
from DRLpractice.UAV.UAVmulti.envs.Env import Env
import time
import numpy as np

MAX_EP_STEPS = 200


def model_eval_MADDPG(maddpg, n_agents):
    eval_env = Env("test")
    eval_env.seed(20)
    observations = eval_env.reset()
    dones = [False for _ in range(n_agents)]
    total_reward = 0.0
    i = 0
    while not np.all(np.array(dones)):
        eval_env.render()
        time.sleep(0.1)
        actions = maddpg.choose_actions(observations=observations)
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
    eval_env.show_path()
    eval_env.close()


def model_eval_MASAC(masac, n_agents):
    eval_env = Env("test")
    eval_env.seed(20)
    observations = eval_env.reset()
    dones = [False for _ in range(n_agents)]
    total_reward = 0.0
    i = 0
    while not np.all(np.array(dones)):
        eval_env.render()
        time.sleep(0.1)
        actions = masac.choose_actions(observations=observations, deterministic=True)
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
    eval_env.show_path()
    eval_env.close()


def model_eval_DDPG(ddpg, n_agents):
    # 适用于DDPG、TD3
    eval_env = Env("test")
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

def model_eval_SAC(sac, n_agents):
    # 适用于DDPG、TD3
    eval_env = Env("test")
    eval_env.seed(20)
    observations = eval_env.reset()
    dones = [False for _ in range(n_agents)]
    total_reward = 0.0
    i = 0
    while not np.all(np.array(dones)):
        eval_env.render()
        time.sleep(0.1)
        actions = [sac[i].choose_action(observations[i], deterministic=True) for i in range(n_agents)]
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
    eval_env.show_path()
    eval_env.close()

if __name__ == "__main__":
    # ######################### MADDPG ###############################
    # state_dim = 15
    # action_dim = 3
    # max_action = 1.0
    # n_agents = 2
    # policy_name = "MADDPG"
    # env_name = "MAStandardEnv"
    # file_name = f"./model_train/MADDPG/{policy_name}_{env_name}"
    # policy = MADDPG(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_MADDPG(policy, n_agents)

    # ########################## MATD3 ###############################
    # state_dim = 15
    # action_dim = 3
    # max_action = 1.0
    # n_agents = 3
    # policy_name = "MATD3"
    # env_name = "MAStandardEnv"
    # file_name = f"./model_train/MATD3/{policy_name}_{env_name}"
    # policy = MATD3(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_MADDPG(policy, n_agents)

    # ########################## MASAC ###############################
    # state_dim = 15
    # action_dim = 3
    # max_action = 1.0
    # n_agents = 2
    # policy_name = "MASAC"
    # env_name = "MAStandardEnv"
    # file_name = f"./model_train/MASAC/{policy_name}_{env_name}"
    # policy = MATD3(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_MADDPG(policy, n_agents)

    # ########################## TD3 ###############################
    state_dim = 15
    action_dim = 3
    max_action = 1.0
    n_agents = 3
    policy_name = "TD3"
    env_name = "MAStandardEnv"
    policy = []
    policy = [TD3(state_dim, action_dim, max_action) for _ in range(n_agents)]
    for i in range(n_agents):
        file_name = f"model_train/TD3/agent_{i}_{policy_name}_{env_name}"
        policy[i].load(file_name)
    # 测试随机选择（非正态分布）
    # random_eval()
    # 测试模型
    model_eval_DDPG(policy, n_agents)

    # ########################## SAC ###############################
    # state_dim = 15
    # action_dim = 3
    # max_action = 1.0
    # n_agents = 3
    # policy_name = "SAC"
    # env_name = "MAStandardEnv"
    # policy = []
    # policy = [SAC(state_dim, action_dim, max_action) for _ in range(n_agents)]
    # for i in range(n_agents):
    #     file_name = f"./model_train/SAC/agent_{i}_{policy_name}_{env_name}"
    #     policy[i].load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_SAC(policy, n_agents)

    ########################## DDPG ###############################
    # state_dim = 15
    # action_dim = 3
    # max_action = 1.0
    # n_agents = 3
    # policy_name = "DDPG"
    # env_name = "MAStandardEnv"
    # policy = []
    # policy = [TD3(state_dim, action_dim, max_action) for _ in range(n_agents)]
    # for i in range(n_agents):
    #     file_name = f"model_train/DDPG/agent_{i}_{policy_name}_{env_name}"
    #     policy[i].load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_DDPG(policy, n_agents)
