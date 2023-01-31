import torch
from DRLpractice.DDPGdemo.DDPG import *
from DRLpractice.DDPGdemo.env import *
from DRLpractice.DDPGdemo.utils import *
import matplotlib.pyplot as plt
import time

# 设定如果100步内没有到达终点则直接done
MAX_EP_STEPS = 100



# 测试随机（均匀分布！后续可以修改为正态分布）情况下的reward
def random_eval():
    eval_env = singleEnv()
    eval_env.seed(10)
    eval_env.render()
    for _ in range(MAX_EP_STEPS):
        action = env.sample_action()
        state, reward, done = eval_env.step(action)
        print(f"当前：   Action:速度增量{action[0]}；  角度增量{action[1] / np.pi * 180}\n"
              f"        State:位置{state[0], state[1]}；  速度{state[2]}；  角度{state[3] / np.pi * 180}\n"
              f"        Reward:{reward}")
        eval_env.render()
        time.sleep(0.001)
        if done:
             break
    eval_env.close()

# 玩1局
def model_eval(policy):
    eval_env = singleEnv()
    eval_env.seed(10)
    eval_env.render()
    state, done = eval_env.reset(), False
    for _ in range(MAX_EP_STEPS):
        action = policy.select_action(
            (np.array(state)-np.array([250, 250, 10, np.pi/4]))/np.array([500, 500, 20, np.pi*2]))
        state, reward, done = eval_env.step(action)
        print(f"当前：   Action:速度增量{action[0]}；  角度增量{action[1] / np.pi * 180}\n"
            f"        State:位置{state[0], state[1]}；  速度{state[2]}；  角度{state[3] / np.pi * 180}\n"
            f"        Reward:{reward}")
        eval_env.render()
        time.sleep(0.001)
        if done:
            break
    eval_env.close()


# 测试
if __name__ == "__main__":
    # 设定环境
    env = singleEnv()
    env.seed(10)
    torch.manual_seed(10)
    np.random.seed(10)

    # 配置model的一些信息
    action_dim = env.action_dim
    state_dim = env.state_dim
    max_action = [1., np.pi / 36]  # 速度增加量最大为 +-1 m/s,最大转弯角设为 +-5 度
    # 载入训练好的模型
    policy = DDPG(state_dim, action_dim, max_action)
    policy.load("./model/ddpg")
    # 测试随机选择（非正态分布）
    random_eval()
    # 测试DDPG模型
    model_eval(policy)

