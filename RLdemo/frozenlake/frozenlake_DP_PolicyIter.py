"""
Solving FrozenLake environment using Policy-Iteration.
Adapted by Bolei Zhou for IERG6130. Originally from Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register


def run_episode(env, policy, gamma=1.0, render=False):  # 重新生成一个初始的状态，然后执行policy，算出得分
    """ Runs an episode and return the total reward """
    obs = env.reset()   # 这里生成了一个初始状态，对应游戏，就是从起点开始走，obs就为0（是一个state值）
    total_reward = 0
    step_idx = 0    # 记录走了多少步骤
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))   # 根据最优策略的指向（三个具体的策略），随机选一个
        # 可以把step理解为蚁群中的轮盘赌、加上往下走的count++；而终止判定的结果(走到终点or死掉)，就是这里的done；count++后的结果就是obs；
        total_reward += (gamma ** step_idx * reward)    # 这里用的gamma是向前传播的，只能设为1，类似于蚁群的信息素衰减因子（有点奇葩）
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
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.env.nS)  # V表初始化为0
    eps = 1e-10     # 收敛判定指标
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):     # 总共16个s，对每一个s
            policy_a = policy[s]    # 当前策略采取的是policy_a这类动作
            # 在这里给出的是明确的上下左右，但是根据这一动作构造的状态转移规则为：例如当前策略为向右，则下一步走向上、右、下的概率分别为0.333
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
            # 对公式结合实际问题理解：对当前s采取policy_a这些动作之后，会以概率p的的可能性转移到s_这个状态，并同时获得r的即使奖励
            # sum(每一个可能状态s_的概率*（所带来的收益折现)
        if np.sum((np.fabs(prev_v - v))) <= eps:
            # value converged
            break
    return v


def policy_iteration(env, gamma=1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  # initialize a random policy
    # 16个states，每1个state对应4个actions，初始化策略表格（16个states，每个state选择上下左右中的1个action）
    max_iterations = 200000
    gamma = 1.0     # 折现因子=1
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)  # 根据游戏环境env和初始策略policy来进行prediction，得到新的V表
        new_policy = extract_policy(old_policy_v, gamma)    # 在新的V表基础上提取新的最优策略
        if np.all(policy == new_policy):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.render()    # 展示初始的游戏环境

    optimal_policy = policy_iteration(env, gamma=1.0)
    scores = evaluate_policy(env, optimal_policy, gamma=1.0)
    print('Average scores = ', scores)
