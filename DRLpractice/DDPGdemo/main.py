import torch
from DRLpractice.DDPGdemo.DDPG import *
from DRLpractice.DDPGdemo.env import *
from DRLpractice.DDPGdemo.utils import *
import matplotlib.pyplot as plt
import time

# 设定如果100步内没有到达终点则直接done
MAX_EP_STEPS = 100


# 测试随机（均匀分布！后续可以修改为正态分布）情况下的reward
def random_eval(eval_episodes=10):
    eval_env = singleEnv()
    # eval_env.render()
    eval_env.seed(100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        # eval_env.render()
        for _ in range(MAX_EP_STEPS):
            action = env.sample_action()
            state, reward, done = eval_env.step(action)
            # eval_env.render()
            # time.sleep(0.001)
            avg_reward += reward
            if done:
                break

    avg_reward /= eval_episodes
    eval_env.close()
    print(f"随机情景下的平均reward为：{avg_reward}")
    return avg_reward


# 测试一次(玩10局)并绘制结果
def model_eval(policy, eval_episodes=10):
    eval_env = singleEnv()
    # eval_env.render()
    eval_env.seed(100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        # eval_env.render()
        for _ in range(MAX_EP_STEPS):
            # DDPG从ReplayBuffer里面拿到的五元组训练参数是经过归一化处理的，所以输进网络的参数得先进行处理
            # 而actor网络输出的值为[-1，1]（tanh），然后又根据max_action进行了调整，所以这里返回的action不需要再进行归一化，直接是可用的
            action = policy.select_action(
                (np.array(state)-np.array([250, 250, 10, np.pi/4]))/np.array([500, 500, 20, np.pi*2]))
            state, reward, done = eval_env.step(action)
            # eval_env.render()
            # time.sleep(0.001)

            avg_reward += reward
            if done:
                break

    avg_reward /= eval_episodes
    eval_env.close()
    # print(f"当前的平均reward为：{avg_reward}")
    return avg_reward


# 每eval_frequent次训练后会进行一次evaluate（一次evaluate会玩10局），将这些测试所得的rewards画出
def eval_result_plot(rewards, eval_frequent):
    plt.plot(rewards)
    plt.ylabel('average reward')
    plt.xlabel(f'evaluate per {eval_frequent} steps')
    plt.title('Evaluation over 10 episodes')
    plt.show()


# 训练
if __name__ == "__main__":
    starttime = time.time()

    # 设定环境
    env = singleEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    max_action = [1., np.pi/36]      # 速度增加量最大为 +-1 m/s,最大转弯角设为 +-5 度
    v_bound = [1, 20]   # 最快20m/s

    # 初始化
    max_size = 1e6     # ReplayBuffer最多能存储的步数
    batch_size = 32     # 原论文中默认64

    # 配置env和model的一些信息
    action_dim = env.action_dim
    state_dim = env.state_dim
    replay_buffer = ReplayBuffer(state_dim, action_dim, int(max_size))
    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action)

    # Evaluate untrained policy
    # evaluations = [random_eval()]
    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0.
    episode_num = 0
    episode_timesteps = 0
    tr_timesteps = 1e6 # 总训练的步数,tr_timesteps和ReplayBuffer的max_size对应，如果要将训练过程中走的每一步都存下来，则max_size设定为和tr_timesteps值相同即可
    start_timesteps = 25e3  # 设置前25e3步不根据policy（即DDPG actor）网络选取动作，积累一定的数据才能进行训练
    eval_freq = 5e3  # 每5e3步就会进行一次测试

    # 开始训练
    for i in range(int(tr_timesteps)):
        episode_timesteps += 1
        # print(f"————————第{i}个episod————————")
        if i < start_timesteps:
            action = env.sample_action()
        else:
            # 在网络输出的action上引入了噪声
            action = (policy.select_action((np.array(state)-np.array([250, 250, 10, np.pi/4]))/np.array([500, 500, 20, np.pi*2]))
                    + np.random.normal(0, [0.05, np.pi / 720], size=action_dim)
            ).clip([-1., -np.pi / 36], max_action)

        # 执行动作
        next_state, reward, done = env.step(action)
        # 打印数据，但是训练效率会因此变慢
        # print(f"当前第{episode_num+1}局的第{episode_timesteps}，总第{i+1}步：\n"
        #       f"        Action:速度增量{action[0]}；  角度增量{action[1] / np.pi * 180}\n"
        #       f"        State:位置{state[0], state[1]}；  速度{state[2]}；  角度{state[3] / np.pi * 180}\n"
        #       f"        Reward:{reward}")

        # 存储
        replay_buffer.add(state, action, next_state, reward, float(done))
        state = next_state
        episode_reward += reward

        if i >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done or episode_timesteps == MAX_EP_STEPS:
            state, done = env.reset(), False
            episode_reward = 0.
            episode_timesteps = 0
            episode_num += 1

        if i >= start_timesteps and (i+1) % eval_freq == 0:
            # print("——————————测试—————————")
            evaluations.append(model_eval(policy))
            # print("—————————————————————")

    policy.save("./model/ddpg")
    eval_result_plot(evaluations, eval_freq)
    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)

