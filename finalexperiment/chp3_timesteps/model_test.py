# coding=utf-8
from finalexperiment.chp3_timesteps.env.eval_env import EvalEnv
from finalexperiment.chp3_timesteps.DDPG import DDPG
from finalexperiment.chp3_timesteps.SAC import SAC
# from finalexperiment.chp3_timesteps.SACLSTM import SACLSTM
from finalexperiment.chp3_timesteps.TD3 import TD3
from finalexperiment.chp3_timesteps.TD3LSTM import TD3LSTM
# from finalexperiment.chp3_timesteps.TD3LSTMATT import TD3LSTMATT
# from finalexperiment.chp3_timesteps.TD3LSTM_buffer import TD3LSTM
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
    # elif policy_name == "SAC_LSTM":
    #     agent = SACLSTM(batch_size, state_dim, action_dim, max_action)
    elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
        agent = TD3LSTM(batch_size, state_dim, action_dim, max_action)
    # elif policy_name == "TD3_LSTM_ATT":
    #     agent = TD3LSTMATT(batch_size, state_dim, action_dim, max_action)

    agent.load(f"./model_train/{policy_name}/{env_name}/{policy_name}")

    path_len = 0.
    path_delay = 0.
    success_path_len = 0.
    success_path_delay = 0.
    eval_env = EvalEnv()
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
        # elif policy_name == "SAC_LSTM":
        #     a, h, c = agent.choose_action(s, h, c, deterministic=True)
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
    # elif policy_name == "SAC_LSTM":
    #     agent = SACLSTM(batch_size, state_dim, action_dim, max_action)
    elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_BUF":
        agent = TD3LSTM(batch_size, state_dim, action_dim, max_action)
    # elif policy_name == "TD3_LSTM_ATT":
    #     agent = TD3LSTMATT(batch_size, state_dim, action_dim, max_action)

    agent.load(f"./model_train/{policy_name}/{env_name}/{policy_name}")

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
        if policy_name == "TD3_LSTM" or policy_name == "SAC_LSTM" \
                or policy_name == "TD3_LSTM_ATT" or policy_name == "TD3_LSTM_BUF":
            h, c = agent.actor.init_hidden_state(batch_size=1, training=False)
        for j in range(MAX_EP_STEPS):
            if policy_name == "TD3" or policy_name == "DDPG":
                a = agent.choose_action(s)
            elif policy_name == "SAC":
                a = agent.choose_action(s, deterministic=True)
            elif policy_name == "TD3_LSTM" or policy_name == "TD3_LSTM_ATT" \
                    or policy_name == "TD3_LSTM_BUF":
                a, h, c = agent.choose_action(s, h, c)
            # elif policy_name == "SAC_LSTM":
            #     a, h, c = agent.choose_action(s, h, c, deterministic=True)

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
    # TD3_LSTM_ATT
    # Dynamic + 随机起点、终点
    # success times:882    collision times:95      path length:7443.463960779172      path delay:375.507074847158
    # success path length:7655.129969504773      success path delay:129.1345213680235
    # Dynamic + 固定测试
    # reward:60.485525957356245    path length:7172.172738899023      path delay:73.75155407812674
    # success path length:7172.172738899023      success path delay:73.75155407812674
    # Static + 随机起点、终点
    # success times:510    collision times:471      path length:6302.076560753471      path delay:1436.7502931608283
    # success path length:7246.546884119162      success path delay:130.76406307020517

    # TD3_LSTM
    # Dynamic + 随机起点、终点
    # success times:929    collision times:57      path length:7521.011073917254      path delay:256.2462204390611
    # success path length:7617.1555361755345      success path delay:122.67653440939473
    # Dynamic + 固定测试
    # reward:63.22234722349804    path length:7218.759861911287      path delay:129.7166576754592
    # success path length:7218.759861911287      success path delay:129.7166576754592
    # Static + 随机起点、终点
    # success times:593    collision times:400      path length:6650.340801593284      path delay:1145.71777361089
    # success path length:7450.293304295547      success path delay:115.88545614590058

    # TD3
    # Dynamic + 随机起点、终点
    # success times:948    collision times:13      path length:7731.136055715205      path delay:189.7597697824522
    # success path length:7799.649423237748      success path delay:131.4520495104695
    # Dynamic + 固定测试
    # reward:56.36615874749219    path length:7322.264281612081      path delay:117.70121036285298
    # success path length:7322.264281612081      success path delay:117.70121036285298
    # Static + 随机起点、终点
    # success times:477    collision times:403      path length:6624.126991460147      path delay:1379.8590127139796
    # success path length:7520.407253991963      success path delay:127.61802121534133

    # SAC
    # Dynamic + 随机起点、终点
    # success times:972    collision times:13      path length:7709.09442002475      path delay:177.01446642388197
    # success path length:7746.921380887056      success path delay:123.86190518093169
    # Dynamic + 固定测试
    # reward:59.361078289122766    path length:7158.679911233542      path delay:131.2519095050616
    # success path length:7158.679911233542      success path delay:131.2519095050616
    # Static + 随机起点、终点
    # success times:456    collision times:455      path length:6453.403734320675      path delay:1548.4195142791032
    # success path length:7356.232525023313      success path delay:121.92662915560862

    # SAC_LSTM
    # Dynamic + 随机起点、终点
    # success times:929    collision times:68      path length:7491.088072635015      path delay:332.8119052765004
    # success path length:7720.734182671938      success path delay:127.64611182365064
    # Dynamic + 固定测试
    # reward:60.58146403708277    path length:7109.547575267581      path delay:118.85946751744379
    # success path length:7109.547575267581      success path delay:118.85946751744379
    # Static + 随机起点、终点
    # success times:536    collision times:447      path length:6643.250147191788      path delay:1215.0789834542163
    # success path length:7483.951626724442      success path delay:128.7784210628361


    ################################################## 论文中##############################
    # TD3：
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:936    collision times:53      path length:7543.2759700825145      path delay:335.43926489515076       success path length:7769.222374264786      success path delay:133.68014600698424
    # Dynamic训练 + Static测试 + 随机：
    # success times:459    collision times:472      path length:6582.889022554074      path delay:1355.7245517194795       success path length:7573.486867145752      success path delay:133.60430655379733

    # LSTM - TD3：
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:925    collision times:47      path length:7583.8422399943165      path delay:210.08442479063902       success path length:7719.622765636671      success path delay:80.42791013985105
    # Dynamic训练 + Static测试 + 随机：
    # success times:452    collision times:524      path length:6281.693246542169      path delay:1535.5276832125485       success path length:7384.1353743438685      success path delay:87.25430183964961

    # R - TD3：
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:945    collision times:52      path length:7524.629065672756      path delay:251.17579846780365       success path length:7669.456087352533      success path delay:105.97213037488983
    # Dynamic训练 + Static测试 + 随机：
    # success times:537    collision times:453      path length:6474.030001823263      path delay:1340.5585570957608       success path length:7472.4595830492135      success path delay:110.89567497412887

    # SAC
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:972    collision times:21      path length:7803.768390604876      path delay:202.36754998898445       success path length:7878.444622188662      success path delay:128.4804503166684
    # Dynamic训练 + Static测试 + 随机：
    # success times:514    collision times:458      path length:6658.022226410264      path delay:1216.771754661763       success path length:7613.356938600244      success path delay:124.91264443960202

    # DDPG
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:521    collision times:58      path length:8414.069348123197      path delay:496.93690025572874       success path length:8284.344332349707      success path delay:136.2512277326196
    # Dynamic训练 + Static测试 + 随机：
    # success times:173    collision times:402      path length:6785.7475760726975      path delay:1647.3613847138413       success path length:7791.88026586382      success path delay:140.33828448181703

    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env_name = "DynamicEnv"
    policy_name = "TD3"

    # 表格数据
    success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model(env_name, policy_name)
    print(f"success times:{success}    collision times:{collision}      "
          f"path length:{path_len}      path delay:{path_delay}       "
          f"success path length:{success_path_len}      success path delay:{success_path_delay}")

    # # 用于画图
    # reward, path_len, path_delay, success_path_len, success_path_delay = run_model(env_name, policy_name)
    # print(f"reward:{reward}    path length:{path_len}      path delay:{path_delay}       "
    #       f"success path length:{success_path_len}      success path delay:{success_path_delay}")



