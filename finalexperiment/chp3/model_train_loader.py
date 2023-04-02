# coding=utf-8
from finalexperiment.chp3.TD3 import TD3
from finalexperiment.chp3.TD3LSTM import TD3LSTM
from finalexperiment.chp3.SAC import SAC
from finalexperiment.chp3.SACLSTM import SACLSTM
from finalexperiment.chp3.DDPG import DDPG
# from finalexperiment.chp3.env.env import Env
from finalexperiment.chp3.env.eval_env import EvalEnv
import torch
import matplotlib.pyplot as plt
import time
from tqdm import *
import numpy as np

# 设定如果100步内没有到达终点则直接done
MAX_EP_STEPS = 200


# 测试随机（均匀分布！后续可以修改为正态分布）情况下的reward
def random_eval():
    eval_env = EvalEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action = eval_env.sample_action()
        s, r, done,_ = eval_env.step(action)
        print(f"currently, the {i + 1} step:\n"
              f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
              f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
              f"           Reward:{r}\n")
        if done:
            eval_env.show_path()
            break
    eval_env.close()


# 测试
def model_eval_DDPG_TD3(policy):
    eval_env = EvalEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    reward = 0.0
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action = policy.choose_action(state)  # Action and the corresponding log probability
        s, r, done,_ = eval_env.step(action)
        # print(f"currently, the {i + 1} step:\n"
        #       f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
        #       f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
        #       f"           Reward:{r}\n")
        state = s
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()

# 测试
def model_eval_SAC(policy):
    eval_env = EvalEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    reward = 0.0
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action = policy.choose_action(state, deterministic=True)  # Action and the corresponding log probability
        s, r, done, _ = eval_env.step(action)
        # print(f"currently, the {i + 1} step:\n"
        #       f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
        #       f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
        #       f"           Reward:{r}\n")
        state = s
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()


def model_eval_SAC_LSTM(agent):
    eval_env = EvalEnv()
    eval_env.seed(20)
    state, done = eval_env.reset(), False
    reward = 0.0
    h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action, h, c = policy.choose_action(state, h, c, deterministic=True)  # Action and the corresponding log probability
        s, r, done,_ = eval_env.step(action)
        # print(f"currently, the {i + 1} step:\n"
        #       f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
        #       f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
        #       f"           Reward:{r}\n")
        state = s
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()

def model_eval_DDPG_TD3_LSTM(agent):
    eval_env = EvalEnv()
    eval_env.seed(20)
    s, done = eval_env.reset(), False
    reward = 0.0
    h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
    for i in range(MAX_EP_STEPS):
        eval_env.render()
        time.sleep(0.1)
        action, h, c = agent.choose_action(s, h, c)  # We do not add noise when evaluating        s, r, done = eval_env.step(action)
        s_, r, done, _ = eval_env.step(action)
        # print(f"currently, the {i + 1} step:\n"
        #       f"           Action: speed {action[0]*5.0, action[1]*5.0, action[2]*0.5}\n"
        #       f"           State: pos {s[0][0]*5000.0, s[0][1]*5000.0, s[0][2]*300.0};   speed {s[0][3]*200, s[0][4]*200.0, s[0][5]*10}\n"
        #       f"           Reward:{r}\n")
        s = s_
        reward += r
        if done:
            break
    print(reward)
    eval_env.show_path()
    eval_env.close()

################################################ 评估模型 ######################################
def test_model_TD3_LSTM(agent):
    success_times = 0
    collision_times = 0
    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.
    for i in tqdm(range(1000)):
        eval_env = EvalEnv()
        s, done = eval_env.reset(), False
        reward = 0.0
        h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
        for i in range(MAX_EP_STEPS):
            action, h, c = agent.choose_action(s, h, c)
            s_, r, done, category = eval_env.step(action)

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

def test_model_DDPG_TD3(agent):
    success_times = 0
    collision_times = 0
    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.
    for i in tqdm(range(1000)):
    # for i in range(1000):
        eval_env = EvalEnv()
        s, done = eval_env.reset(), False
        reward = 0.0
        for i in range(MAX_EP_STEPS):
            action = agent.choose_action(s)
            s_, r, done, category = eval_env.step(action)

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

def test_model_SAC(agent):
    success_times = 0
    collision_times = 0
    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.
    for i in tqdm(range(1000)):
        eval_env = EvalEnv()
        s, done = eval_env.reset(), False
        reward = 0.0
        for i in range(MAX_EP_STEPS):
            action = agent.choose_action(s, deterministic=True)
            s_, r, done, category = eval_env.step(action)

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

def test_model_SAC_LSTM(agent):
    success_times = 0
    collision_times = 0
    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.
    for i in tqdm(range(1000)):
        eval_env = EvalEnv()
        s, done = eval_env.reset(), False
        reward = 0.0
        h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
        for i in range(MAX_EP_STEPS):
            action, h, c = agent.choose_action(s, h, c, deterministic=True)
            s_, r, done, category = eval_env.step(action)

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

    # # ####################################  载入DDPG  ##########################################
    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # policy_name = "DDPG"
    # env_name = "DynamicEnv"
    # file_name = f"./model_train/DDPG/{policy_name}_{env_name}"
    # policy = DDPG(state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_DDPG_TD3(policy)


    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ####################################  载入TD3  ##########################################
    # Standard的模型 + Standard Env
    # success times:930  collision times:50  path length:7644.856307455912  path delay:264.31697810769447
    # Dynamic的模型 + Dynamic Env
    # success times:982    collision times:16      path length:7648.922173642492      path delay:176.41172030321206
    # success times:977    collision times:17      path length:7698.150724735572      path delay:186.61103387822766

    # success times:978    collision times:14      path length:7628.423588430769      path delay:164.9958592444903
    # success path length:7692.553931602223      success path delay:108.69146900188458


    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # policy_name = "TD3"
    # env_name = "DynamicEnv"
    # file_name = f"./model_train/TD3/{policy_name}_{env_name}"
    # policy = TD3(state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_DDPG_TD3(policy)
    # success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model_DDPG_TD3(policy)
    # print(f"success times:{success}    collision times:{collision}      "
    #       f"path length:{path_len}      path delay:{path_delay}       "
    #       f"success path length:{success_path_len}      success path delay:{success_path_delay}")

    # # ####################################  载入SAC  ##########################################
    # Standard的模型 + Standard Env
    # success times:977   collision times:22   path length:7610.647886955501  path delay:212.09205858580643
    # Dynamic的模型 + Standard Env
    # success times:511    collision times:434      path length:6418.474042761665      path delay:1448.197954104928
    # Dynamic的模型 + Dynamic Env
    # success times:977    collision times:14      path length:7640.397935736107      path delay:195.55482561756963

    # success times:965    collision times:19      path length:7623.457616670847      path delay:211.59848901278284
    # success path length:7711.147735715194      success path delay:131.75976839659188

    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # policy_name = "SAC"
    # env_name = "DynamicEnv"
    # file_name = f"./model_train/SAC/{policy_name}_{env_name}"
    # policy = SAC(state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_SAC(policy)
    # success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model_SAC(policy)
    # print(f"success times:{success}    collision times:{collision}      "
    #       f"path length:{path_len}      path delay:{path_delay}       "
    #       f"success path length:{success_path_len}      success path delay:{success_path_delay}")


    # ####################################  载入TD3-LSTM  ##########################################
    # Dynamic的模型 + Standard Env
    # success times:507    collision times:471      path length:6360.664407059632      path delay:1459.333958778304
    # Dynamic的模型 + Dynamic Env
    # success times:923    collision times:66      path length:7523.251134755234      path delay:290.1445818551986

    # success times:928    collision times:71      path length:7483.190411012691      path delay:281.7398567200422
    # success path length:7661.995028390355      success path delay:109.75274522970281

    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # batch_size = 1
    # policy_name = "TD3LSTM"
    # env_name = "DynamicEnv"
    # file_name = f"./model_train/TD3LSTM/{policy_name}_{env_name}"
    # policy = TD3LSTM(batch_size, state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_DDPG_TD3_LSTM(policy)
    # success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model_TD3_LSTM(policy)
    # print(f"success times:{success}    collision times:{collision}      "
    #       f"path length:{path_len}      path delay:{path_delay}       "
    #       f"success path length:{success_path_len}      success path delay:{success_path_delay}")

    # # ####################################  载入TD3-LSTM-ATT  ##########################################
    # Dynamic的模型 + Standard Env
    # success times:478    collision times:445      path length:6969.035004772874      path delay:1357.9711049756002
    # Dynamic的模型 + Dynamic Env
    # success times:847    collision times:112      path length:7429.5965885829855      path delay:373.60923318127607
    # success times:860    collision times:92      path length:7549.6253023961435      path delay:327.24598617060605
    # success times:871    collision times:94      path length:7472.083084567646      path delay:315.24572250736304
    # success path length:7622.702968755556      success path delay:136.29190427796107

    # seed=20
    # success times:1000    collision times:0      path length:7253.473938958011      path delay:112.61692133102034
    # success path length:7253.473938958011      success path delay:112.61692133102034

    state_dim = 12
    action_dim = 3
    max_action = 1.0
    batch_size = 1
    policy_name = "TD3LSTMATT"
    env_name = "DynamicEnv"
    file_name = f"./model_train/TD3LSTMATT/{policy_name}_{env_name}"
    policy = TD3LSTM(batch_size, state_dim, action_dim, max_action)
    policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # 测试模型
    # model_eval_DDPG_TD3_LSTM(policy)
    success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model_TD3_LSTM(policy)
    print(f"success times:{success}    collision times:{collision}      "
          f"path length:{path_len}      path delay:{path_delay}       "
          f"success path length:{success_path_len}      success path delay:{success_path_delay}")



    # # ####################################  载入SAC-LSTM ##########################################
    # Dynamic的模型 + Standard Env
    # success times:529    collision times:426      path length:6589.532223038095      path delay:1254.9163721126235
    # Dynamic的模型 + Dynamic Env
    # success times:926    collision times:62      path length:7458.750584833313      path delay:303.87656087294295

    # success times:934    collision times:54      path length:7560.589464288793      path delay:263.58853075570283
    # success path length:7706.328112141063      success path delay:126.43095616774302
    # success times:936    collision times:48      path length:7534.210881280407      path delay:257.92992586000173
    # success path length:7672.1677287263365      success path delay:125.12957521136036

    # seed = 20
    # success times:937    collision times:53      path length:7486.596177659465      path delay:289.7321863928938
    # success path length:7648.591779321831      success path delay:127.2485516645541

    # state_dim = 12
    # action_dim = 3
    # max_action = 1.0
    # batch_size = 1
    # policy_name = "SACLSTM"
    # env_name = "DynamicEnv"
    # file_name = f"./model_train/SACLSTM/{policy_name}_{env_name}"
    # policy = SACLSTM(batch_size, state_dim, action_dim, max_action)
    # policy.load(file_name)
    # # 测试随机选择（非正态分布）
    # # random_eval()
    # # # 测试模型
    # model_eval_SAC_LSTM(policy)
    # success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model_SAC_LSTM(policy)
    # print(f"success times:{success}    collision times:{collision}      "
    #       f"path length:{path_len}      path delay:{path_delay}       "
    #       f"success path length:{success_path_len}      success path delay:{success_path_delay}")


