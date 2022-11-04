"""
Model-free Control for OpenAI FrozenLake env (https://gym.openai.com/envs/FrozenLake-v0/)
Bolei Zhou for IERG6130 course example
"""
import gym
import numpy as np
from gym.envs.registration import register

no_slippery = True  # ！！！！！！DP的两个算法中，这里就是False！！！也就是给出向左的策略，向上下左概率都为1/3，而True的时候，说向左就向左！
render_last = True  # whether to visualize the last episode in testing

# -- hyperparameters--
num_epis_train = 10000
num_iter = 100
learning_rate = 0.01
discount = 0.8
eps = 0.3

if no_slippery == True:
    # the simplified frozen lake without slippery (so the transition is deterministic)
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=1000,
        reward_threshold=0.78,  # optimum = .8196
    )
    env = gym.make('FrozenLakeNotSlippery-v0')
else:
    # the standard slippery frozen lake
    env = gym.make('FrozenLake-v1')

q_learning_table = np.zeros([env.observation_space.n, env.action_space.n])  # 生成states行，actions列的array（16x4）

# -- training the agent ----
for epis in range(num_epis_train):  # 对每一个episode
    state = env.reset()  # episode游戏环境初始化，对应游戏，就是从起点开始走，state就为0
    for iter in range(num_iter):
        if np.random.uniform(0, 1) < eps:  # 有eps（30%）的可能性随机选择动作（exploration）
            action = np.random.choice(env.action_space.n)
        else:  # 否则，贪心选择当前最优q_learning_table值的动作（70%的概率，exploitation）
            action = np.argmax(q_learning_table[state, :])
        state_new, reward, done, _ = env.step(action)  # 开始玩
        # 可以把step理解为蚁群中的轮盘赌、加上往下走的count++；终止判定的结果(走到终点or死掉)，就是done；count++后的结果就是state_new；
        q_learning_table[state, action] = q_learning_table[state, action] + learning_rate * (
                reward + discount * np.max(q_learning_table[state_new, :]) - q_learning_table[state, action])
        state = state_new
        if done: break

print(np.argmax(q_learning_table, axis=1))  # 输出当前最优策略
# numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值。
# n维的数组的 axis 可以取值从 0 到 n-1，其对应的括号层数为从最外层向内递进，相当于确定观察角度为state，一个state对应的4个actions中取最优
print(np.around(q_learning_table, 6))  # 输出当前q_learning_table

if no_slippery == True:
    print('---Frozenlake without slippery move-----')
else:
    print('---Standard frozenlake------------------')

# visualize no uncertainty
# 测试！！！！！用于评价找到的最优策略，即 np.argmax(q_learning_table[s, :])
num_episode = 500  # 玩500轮（0-499）
rewards = 0
for epi in range(num_episode):
    s = env.reset()
    for _ in range(100):  # 最多走100步（0-99）
        action = np.argmax(q_learning_table[s, :])
        state_new, reward_episode, done_episode, _ = env.step(action)
        if epi == num_episode - 1 and render_last:  # 只可视化最后玩的那一轮的过程（每一步直到done，最多100步）
            env.render()
        s = state_new
        if done_episode:
            if reward_episode == 1:
                rewards += 1
            break

print('---Success rate=%.3f' % (rewards * 1.0 / num_episode))
print('-------------------------------')
