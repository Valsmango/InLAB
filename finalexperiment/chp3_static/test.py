# coding=utf-8
from finalexperiment.chp3_static.env.eval_env import EvalEnv
from finalexperiment.chp3_static.DDPG import DDPG
from finalexperiment.chp3_static.SAC import SAC
# from finalexperiment.chp3_timesteps.SACLSTM import SACLSTM
from finalexperiment.chp3_static.TD3 import TD3
from finalexperiment.chp3_static.TD3_LSTM import TD3LSTM
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
    # success times:936    collision times:53      path length:7543.2759700825145      path delay:335.43926489515076       success path length:7769.222374264786
    # success path delay:133.68014600698424
    # Dynamic训练 + Static测试 + 随机：
    # success times:459    collision times:472      path length:6582.889022554074      path delay:1355.7245517194795       success path length:7573.486867145752      success path delay:133.60430655379733
    # Static训练 + Static测试 + 随机：
    # success times:942    collision times:40      path length:7634.614198292801      path delay:230.41147760816554       success path length:7710.611473400483
    # success path delay:137.1247671265676
    # success times:749    collision times:51      path length:7871.393792950669      path delay:265.03581157502424       success path length:7803.9291938912875
    # success path delay:116.98944244920528
    # Static训练 + Dynamic测试 + 随机：
    # success times:431    collision times:550      path length:6040.883429360787      path delay:1771.8200218627976       success path length:7753.984778304164      success path delay:138.45528554866257
    # success times:308    collision times:609      path length:6083.682298579209      path delay:1775.3366611796314       success path length:7951.151612367345      success path delay:119.00502347557206

    # LSTM - TD3：
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:925    collision times:47      path length:7583.8422399943165      path delay:210.08442479063902       success path length:7719.622765636671
    # success path delay:80.42791013985105
    # Dynamic训练 + Static测试 + 随机：
    # success times:452    collision times:524      path length:6281.693246542169      path delay:1535.5276832125485       success path length:7384.1353743438685      success path delay:87.25430183964961
    # Static训练 + Static测试 + 随机：
    # success times:932    collision times:57      path length:7534.251549169027      path delay:212.9120890573551       success path length:7658.345148241775
    # success path delay:90.80545253672337
    # success times:893    collision times:94      path length:7452.924725308581      path delay:264.387630015863       success path length:7594.9271056373545
    # success path delay:117.93450231044274
    # Static训练 + Dynamic测试 + 随机：
    # success times:429    collision times:564      path length:5923.16076383869      path delay:1741.447722702213       success path length:7805.952055550429      success path delay:96.33097793600007
    # success times:499    collision times:499      path length:6217.730377397284      path delay:1507.796474290977       success path length:7828.682891628203      success path delay:122.88323948694155

    # R - TD3：
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:945    collision times:52      path length:7524.629065672756      path delay:251.17579846780365       success path length:7669.456087352533
    # success path delay:105.97213037488983
    # Dynamic训练 + Static测试 + 随机：
    # success times:537    collision times:453      path length:6474.030001823263      path delay:1340.5585570957608       success path length:7472.4595830492135
    # success path delay:110.89567497412887
    # Static训练 + Static测试 + 随机：
    # success times:938    collision times:27      path length:7558.6502295968285      path delay:181.17348757765362       success path length:7589.501187783916
    # success path delay:104.25853164364355
    # Static训练 + Dynamic测试 + 随机：
    # success times:410    collision times:560      path length:6138.732017989524      path delay:1690.3961961740922       success path length:7775.184937910176      success path delay:104.69523809425684


    # SAC
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:972    collision times:21      path length:7803.768390604876      path delay:202.36754998898445       success path length:7878.444622188662
    # success path delay:128.4804503166684
    # Dynamic训练 + Static测试 + 随机：
    # success times:514    collision times:458      path length:6658.022226410264      path delay:1216.771754661763       success path length:7613.356938600244      success path delay:124.91264443960202
    # Static训练 + Static测试 + 随机：
    # success times:973    collision times:27      path length:7527.318507015221      path delay:227.08555993001238       success path length:7587.804076587679      success path delay:138.7749221568702
    # success times:963    collision times:26      path length:7617.398852607093      path delay:217.9793852541838       success path length:7661.784357960565
    # success path delay:139.9833199243658
    # Static训练 + Dynamic测试 + 随机：
    # success times:381    collision times:614      path length:5892.210008413976      path delay:1883.7124827652817       success path length:7827.891053390512      success path delay:138.59775062724916
    # success times:456    collision times:503      path length:6087.53360423134      path delay:1671.2557149801353       success path length:7677.29294177326      success path delay:127.0739922614362

    # DDPG
    # Dynamic训练 + Dynamic测试 + 随机：
    # success times:521    collision times:58      path length:8414.069348123197      path delay:496.93690025572874       success path length:8284.344332349707
    # success path delay:136.2512277326196
    # Dynamic训练 + Static测试 + 随机：
    # success times:173    collision times:402      path length:6785.7475760726975      path delay:1647.3613847138413       success path length:7791.88026586382      success path delay:140.33828448181703
    # Static训练 + Static测试 + 随机：
    # success times:386    collision times:220      path length:8256.746005274721      path delay:704.87508565761       success path length:8502.874190867586      success path delay:123.89648378576517
    # success times:522    collision times:43      path length:8758.942316305616      path delay:333.4173158504665       success path length:8278.967972659902
    # success path delay:128.09084236295436
    # Static训练 + Dynamic测试 + 随机：
    # success times:202    collision times:464      path length:6871.794270164033      path delay:1620.7151803101206       success path length:8759.57183913126      success path delay:124.41204291819587
    # success times:382    collision times:450      path length:7131.533755505498      path delay:1515.5731303789935       success path length:8254.849211828005      success path delay:125.78914638905549


    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env_name = "StaticEnv_2e6"
    policy_name = "SAC"

    # 表格数据
    # success, collision, path_len, path_delay, success_path_len, success_path_delay = test_model(env_name, policy_name)
    # print(f"success times:{success}    collision times:{collision}      "
    #       f"path length:{path_len}      path delay:{path_delay}       "
    #       f"success path length:{success_path_len}      success path delay:{success_path_delay}")

    # # 用于画图
    reward, path_len, path_delay, success_path_len, success_path_delay = run_model(env_name, policy_name)
    print(f"reward:{reward}    path length:{path_len}      path delay:{path_delay}       "
          f"success path length:{success_path_len}      success path delay:{success_path_delay}")



