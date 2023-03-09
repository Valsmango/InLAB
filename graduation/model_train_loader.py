# coding=utf-8
from graduation.PPO import *
from graduation.TD3 import *
from graduation.TD3LSTM import *
from graduation.SAC import *
from graduation.SACLSTM import *
from graduation.DDPG import *
import torch
import matplotlib.pyplot as plt
import time

# 设定如果100步内没有到达终点则直接done
MAX_EP_STEPS = 200


# 测试随机（均匀分布！后续可以修改为正态分布）情况下的reward
def random_eval():
    eval_env = StandardEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action = eval_env.sample_action()
        s, r, done = eval_env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
              f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
              f"           Reward:{r}\n")
        if done:
            eval_env.show_path()
            break
    eval_env.close()

# 玩1局
def model_eval_PPO(policy, args):
    eval_env = StandardEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    reward = 0.0
    state_norm = Normalization(shape=args.state_dim)
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        if args.use_state_norm:
            state = state_norm(state)
        a, a_logprob = policy.choose_action(state)  # Action and the corresponding log probability
        if args.policy_dist == "Beta":
            action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
        else:
            action = a
        s, r, done = eval_env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
              f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
              f"           Reward:{r}\n")
        state = s
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()


# 测试
def model_eval_DDPG_TD3(policy):
    eval_env = StandardEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    reward = 0.0
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action = policy.choose_action(state)  # Action and the corresponding log probability
        s, r, done = eval_env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
              f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
              f"           Reward:{r}\n")
        state = s
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()

# 测试
def model_eval_SAC(policy):
    eval_env = StandardEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    reward = 0.0
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action = policy.choose_action(state, deterministic=True)  # Action and the corresponding log probability
        s, r, done = eval_env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
              f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
              f"           Reward:{r}\n")
        state = s
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()


def model_eval_SAC_LSTM(agent):
    eval_env = StandardEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    reward = 0.0
    h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action, h, c = policy.choose_action(state, h, c, deterministic=True)  # Action and the corresponding log probability
        s, r, done = eval_env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
              f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
              f"           Reward:{r}\n")
        state = s
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()

def model_eval_DDPG_TD3_LSTM(agent):
    eval_env = StandardEnv()
    eval_env.seed(20)
    s, done = eval_env.reset(), False
    reward = 0.0
    h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action, h, c = agent.choose_action(s, h, c)  # We do not add noise when evaluating        s, r, done = eval_env.step(action)
        s_, r, done = eval_env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
              f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
              f"           Reward:{r}\n")
        s = s_
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()


if __name__ == "__main__":

    # ####################################  载入PPO  ##########################################
    # # 载入训练好的模型
    # policy_name = "PPO"
    # env_name = "StandardEnv"
    # seed_num = 10
    # file_name = f"./model_train/PPO/{policy_name}_{env_name}"
    #
    # parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    # parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    # parser.add_argument("--evaluate_freq", type=float, default=5e3,
    #                     help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    # parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
    # parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    # parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    # parser.add_argument("--hidden_width", type=int, default=256,
    #                     help="The number of neurons in hidden layers of the neural network")
    # parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    # parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    # parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    # parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    # parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    # parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    # parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    # parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    # parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    # parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    # parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    # parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    # parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    # parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    # parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    # parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    # args = parser.parse_args()
    # args.state_dim = 12
    # args.action_dim = 3
    # args.max_action = 1.0
    # args.max_episode_steps = 200  # Maximum number of steps per episode
    # policy = PPO_continuous(args)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_PPO(policy, args)

    # ####################################  载入TD3  ##########################################
    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # policy_name = "TD3"
    # env_name = "StandardEnv"
    # file_name = f"./model_train/TD3/{policy_name}_{env_name}"
    # policy = TD3(state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_DDPG_TD3(policy)

    # # ####################################  载入SAC  ##########################################
    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # policy_name = "SAC"
    # env_name = "StandardEnv"
    # file_name = f"./model_train/SAC/{policy_name}_{env_name}"
    # policy = SAC(state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_SAC(policy)
    #
    # # ####################################  载入DDPG  ##########################################
    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # policy_name = "DDPG"
    # env_name = "StandardEnv"
    # file_name = f"./model_train/DDPG/{policy_name}_{env_name}"
    # policy = DDPG(state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_DDPG_TD3(policy)

    # ####################################  载入TD3-LSTM  ##########################################
    state_dim = 12
    action_dim = 3
    max_action = 1.0
    batch_size = 1
    policy_name = "TD3LSTM"
    env_name = "StandardEnv"
    file_name = f"./model_train/TD3LSTM/{policy_name}_{env_name}"
    policy = TD3LSTM(batch_size, state_dim, action_dim, max_action)
    policy.load(file_name)
    # 测试随机选择（非正态分布）
    # random_eval()
    # 测试模型
    model_eval_DDPG_TD3_LSTM(policy)

    # # ####################################  载入SAC-LSTM ##########################################
    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # batch_size = 1
    # policy_name = "SACLSTM"
    # env_name = "StandardEnv"
    # file_name = f"./model_train/SACLSTM/{policy_name}_{env_name}"
    # policy = SACLSTM(batch_size, state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_SAC_LSTM(policy)