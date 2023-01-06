'''
some tricks:
https://zhuanlan.zhihu.com/p/512327050


references:
https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/9.PPO-discrete-RNN
'''

import collections
import itertools
import math
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter

import DRLpractice.GymAlgorithms.plotUtils.PlotUtils as pltutil
'''
tensorboard使用：
    https://blog.51cto.com/u_15279692/5520767


tensorboard --logdir=./runs/tensorboard/ppo
'''

import time

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, batch_size):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []
        self.batch_size = batch_size

    def put(self, s, a, r, done, p, v):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(done)
        self.prob.append(p)
        self.value.append(v)

    def sample(self):
        '''
        待填充，选出samples
        '''
        batch_step = np.arange(0, len(self.done), self.batch_size)  # np.arange(start = 0, stop = len, step = batch_sz) 即： 【0， batch_sz, batch_sz*2, ... < stop】
        indicies = np.arange(len(self.done), dtype=np.int64)  # np.arange(start = 0, stop = len, step = 1)
        np.random.shuffle(indicies)     # 将indicies弄成乱序
        batches = [indicies[i:i+self.batch_size] for i in batch_step]
        return self.state, self.action, self.reward, self.done, self.prob, self.value, batches

    def clear(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []

    def __len__(self) -> int:
        return len(self.done)


class CriticNetwork(nn.Module):
    '''
    Critic输入为：state
    输出为： value(state)的估计值
    '''
    def __init__(self, state_space, output_size=1):
        # 继承nn.Module：
        super(CriticNetwork, self).__init__()

        self.hidden_space = 64
        self.state_space = state_space
        self.output_size = output_size

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.output_size)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)


class ActorNetwork(nn.Module):
    '''
    Actror输入为：state
    输出为： action概率分布

        随机性策略，相当于训练过程中的探索exploration，对应的操作就是采样
        强化学习中有两种常见的随机性策略：
            分类策略categorical policy：用来离散动作空间问题
            分类策略可以看做离散动作空间的分类器 —— 输入是观测，经过一些神经网络层，
            输出每个动作的logits，最后用softmax转化为每一个动作的概率probability
            给定每一个动作的概率，可以使用Pytorch中的一些采样函数进行采样，比如Categorical distributions in PyTorch，torch.multinomial

            对角高斯策略diagnoal Gaussian policy：用于连续动作空间问题
            输出维度为action_dim，意义是每个action的高斯策略的均值
            另外，Actor网络还有action_dim个标准差参数，这样在输入一个state后，每个动作都对应一个一维的高斯分布。
            关于噪声向量可以通过 torch.normal来得到。同样，也可以通过构建分布对象来生成采样结果， torch.distributions.Normal，后者的优势是这些对象也可以用来计算对数似然。
    '''
    def __init__(self, state_space, action_space):
        super(ActorNetwork, self).__init__()

        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.softmax(self.Linear3(x), dim=-1)
        x = Categorical(x)
        return x


class DiscretePPO:
    def __init__(self, batch_size, state_space, action_space, clip,
                 actor_lr, critic_lr, epochs, gamma, gae_lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.clip = clip

        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.batch_size)
        self.actor = ActorNetwork(state_space=self.state_space, action_space=self.action_space)
        self.critic = CriticNetwork(state_space=self.state_space)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def sample_action(self, state):
        '''
        state: 输入时的格式应当为todevice了的tensor
        为什么不用 epsilon greedy ？？？？？？？？？？？？？？？？
        '''
        state = np.array([state])  # 先转成数组再转tensor更高效
        state = torch.tensor(state, dtype=torch.float).to(device)
        distribution = self.actor(state)
        action = distribution.sample()
        probs = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()

        value = self.critic(state)
        value = torch.squeeze(value).item()

        return action, probs, value

    def update(self):
        '''
        玩完一整把游戏才能进行多次更新，和DQN这种膈几步就更新一次的不一样
        '''
        for _ in range(self.epochs):
            state, action, reward, done, old_prob, value, batches = self.memory.sample()
            # GAE
            advantage = np.zeros(len(reward), dtype=np.float32)
            for i in range(len(reward) - 1):
                discount = 1
                a_i = 0
                for k in range(i, len(reward) - 1):
                    a_i += discount * (reward[k] + self.gamma * value[k+1] * (1 - int(done[k])) - value[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[i] = a_i
            advantage = torch.tensor(advantage).to(device)
            value = torch.tensor(value).to(device)
            for batch in batches:
                batch_state = torch.tensor(np.array(state)[batch], dtype=torch.float).to(device)
                batch_old_prob = torch.tensor(np.array(old_prob)[batch]).to(device)
                batch_action = torch.tensor(np.array(action)[batch]).to(device)
                # 计算batch_new_prob
                distribution = self.actor(batch_state)
                batch_new_prob = distribution.log_prob(batch_action)    # ????????????
                # 计算actor的loss，即核心的更新公式
                prob_ratio = batch_new_prob.exp() / batch_old_prob.exp()
                weighted_prob = prob_ratio * advantage[batch]
                weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage[batch]
                actor_loss = -torch.min(weighted_prob, weighted_clipped_prob).mean()
                # 更新actor网络
                actor_optimizer = optim.Adam(self.actor.parameters(), self.actor_lr)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                # 计算critic的loss
                batch_critic_value = torch.squeeze(self.critic(batch_state))
                batch_real_value = advantage[batch] + value[batch]
                critic_loss = F.mse_loss(batch_real_value,
                                         batch_critic_value)  # critic_loss = (batch_real_value - batch_critic_value) ** 2    # critic_loss = critic_loss.mean()
                # 更新critic网络
                critic_optimizer = optim.Adam(self.critic.parameters(), self.critic_lr)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
        self.memory.clear()

    def test_model(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def to_device(self):
        self.actor.to(device)
        self.critic.to(device)


if __name__ == "__main__":
    start_time = time.time()

    # Env parameters
    model_name = "PPO"
    env_name = "CartPole-v0"
    seed = 10
    save_model_flag = False
    # writer = SummaryWriter("./runs/tensorboard/ppo")

    # Set gym environment
    env = gym.make(env_name)
    n_actions = env.action_space.n
    try:
        n_states = env.observation_space.n
    except AttributeError:
        n_states = env.observation_space.shape[0]

    # Set seeds
    env.seed(seed)
    # env.action_space.seed(seed)     # 这个是TD3中提到的seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # save results
    file_name = f"{model_name}_{env_name}_{seed}"
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model_flag and not os.path.exists("./models"):
        os.makedirs("./models")

    # model parameters
    # 适当设置batch size，过小的经验池容量和batchsize导致收敛到局部最优，结果呈现震荡形式
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-3
    batch_size = 8
    max_episodes = 200
    clip = 0.2
    gamma = 0.99
    gae_lambda = 0.92
    step_count = 0
    epochs = 10
    update_frequent = 20

    # Initiate the network and set the optimizer
    env.reset()
    model = DiscretePPO(batch_size=batch_size, state_space=n_states, action_space=n_actions, clip=clip,
                        actor_lr=actor_learning_rate, critic_lr=critic_learning_rate,
                        epochs=epochs, gamma=gamma, gae_lambda=gae_lambda)
    model.to_device()

    # output the reward
    rewards = []
    ma_rewards = []
    print_per_iter = 20     # 每玩1把游戏进行一次结果输出
    score = 0
    score_sum = 0.0

    # 开始训练
    for i in range(max_episodes):
        # Initialize the environment and state
        state = env.reset()
        done = None
        while not done:
            # Select and perform an action
            action, prob, val = model.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            # next_state = next_state[::2]
            # reward是一个float格式的数值
            score += reward
            score_sum += reward
            done_mask = 0.0 if done else 1.0
            model.memory.put(s=state, a=action, r=reward, done=done_mask, p=prob, v=val)

            # Move to the next state
            state = next_state
            step_count += 1

            # Perform one step of the optimization
            if step_count % update_frequent == 0:
                model.update()

        rewards.append(score)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * score)
        else:
            ma_rewards.append(score)
        # 隔一段时间，输出一次训练的结果
        if i % print_per_iter == 0 and i != 0:
            print("n_episode :{}, score : {:.1f}".format(i, score))
        # writer.add_scalar("rewards", score, i + 1)
        score = 0
    # save the model
    if save_model_flag:
        pass

    end_time = time.time()
    print("-----------------------------------------")
    print(f"运行时间为：{end_time - start_time:.2f}s")
    print("-----------------------------------------")
    pltutil.plot_certain_training_rewards(rewards, "PPO", "PPO-CartPole rewards")
