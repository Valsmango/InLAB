import numpy as np
import gym

""" 
evaluate_policy + run_episode are used to test the optimal policy
"""


def run_episode(env, policy, gamma=1.0, render=False):  # 重新生成一个初始的状态，然后执行policy，算出得分
    """ Runs an episode and return the total reward """
    obs = env.reset()  # 这里生成了一个初始状态，对应游戏，就是从起点开始走，obs就为0（是一个state值）
    total_reward = 0
    step_idx = 0  # 记录走了多少步骤
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))  # 根据最优策略的指向（三个具体的策略），随机选一个
        # 可以把step理解为蚁群中的轮盘赌、加上往下走的count++；而终止判定的结果(走到终点or死掉)，就是这里的done；count++后的结果就是obs；
        total_reward += (gamma ** step_idx * reward)  # 这里用的gamma是向前传播的，只能设为1，类似于蚁群的信息素衰减因子（有点奇葩）
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])     # 最优策略的提取，实质就是找maxQ
            # for next_sr in env.env.P[s][a]:
            #     # next_sr is a tuple of (probability, next state, reward, done)
            #     p, s_, r, _ = next_sr
            #     q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma=1.0):
    max_iterations = 200000
    eps = 1e-20
    # Policy Iteration需要先初始化一个policy，V表在prediction中初始化，而value iteration则需要在这里初始化V表
    v = np.zeros(env.env.nS)
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            # 关键的区别就在于a挨着看了一遍，总共就有4个值，每个值是对应三个位的下一步s_的v的折现之和
            v[s] = max(q_sa)
        if np.sum(np.fabs(prev_v - v)) <= eps:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return v


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.render()  # 展示初始的游戏环境
    gamma = 1.0
    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Average scores = ', policy_score)
