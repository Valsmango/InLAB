# coding=utf-8
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter

from DRLpractice.UAV.UAVsingle.replaybuffer import PPOReplayBuffer

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class CriticNetwork(nn.Module):
    '''
    Critic输入为：state
    输出为： value(state)的估计值
    '''
    def __init__(self, state_space, output_size=1):
        super(CriticNetwork, self).__init__()

        self.hidden_space = 64
        self.state_space = state_space
        self.output_size = output_size

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.output_size)
        orthogonal_init(self.Linear1)
        orthogonal_init(self.Linear2)
        orthogonal_init(self.Linear3)

    def forward(self, x):
        x = torch.tanh(self.Linear1(x))
        x = torch.tanh(self.Linear2(x))
        return self.Linear3(x)


class ActorNetwork(nn.Module):
    '''
    Actror输入为：state
    输出为： action概率分布
    '''
    def __init__(self, state_space, action_space, max_action):
        super(ActorNetwork, self).__init__()

        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)  # linear3输出均值mu
        self.max_action = torch.Tensor(max_action).to(device)
        self.log_sigma = nn.Parameter(torch.zeros(1, action_space))

        orthogonal_init(self.Linear1)
        orthogonal_init(self.Linear2)
        orthogonal_init(self.Linear3, gain=0.01)

    def forward(self, x):
        x = torch.tanh(self.Linear1(x))  # 有的用的tanh进行激活
        x = torch.tanh(self.Linear2(x))
        mu = self.max_action * torch.tanh(self.Linear3(x))   # 连续动作空间不需要softmax，而是要改进成输出一个均值mu
        log_sigma = self.log_sigma + torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return {'mu': mu, 'sigma': sigma}


class PPO:
    def __init__(self, batch_size, state_space, action_space, clip,
                 actor_lr, critic_lr, epochs, gamma, gae_lambda, max_action):
        self.state_space = state_space
        self.action_space = action_space
        self.clip = clip

        self.batch_size = batch_size
        self.memory = PPOReplayBuffer()
        self.actor = ActorNetwork(state_space=self.state_space, action_space=self.action_space, max_action=max_action).to(device)
        self.critic = CriticNetwork(state_space=self.state_space).to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        action_std_init = 1
        self.action_var = torch.full((3,), action_std_init * action_std_init).to(device)

    def select_action(self, state):
        '''
        state: 输入时的格式应当为todevice了的tensor
        为什么不用 epsilon greedy ？？？？？？？？？？？？？？？？
        '''
        # state = np.array([state])  # 先转成数组再转tensor更高效
        # state = torch.tensor(state, dtype=torch.float).to(device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        logit = self.actor(state)
        dist = Normal(logit['mu'], logit['sigma'])
        # 将 ``action_space`` 个高斯分布转义为一个有着对角协方差矩阵的多维高斯分布。
        # 并保证高斯分布中，每一维之间都是互相独立的（因为协方差矩阵是对角矩阵）
        dist = Independent(dist, 1)
        # 为一个 batch 里的每个样本采样一个维度为 ``action_shape`` 的连续动作，并返回它
        action = dist.sample()
        probs = dist.log_prob(action)
        # action = torch.squeeze(action).item()  # 因为action还是size（[batch_size或1 , 2]）维度的数据，实则只需要size（[2]）的数据
        # probs = torch.squeeze(probs).item()

        value = self.critic(state)
        # value = torch.squeeze(value).item()  # 因为value还是size（[batch_size或1 , 1]）维度的数据，实则只需要size（[1]）的数据

        return action.cpu().data.numpy().flatten(), probs.cpu().data.numpy().flatten(), value.cpu().data.numpy().flatten()

    def update(self):
        '''
        玩完一整把游戏才能进行多次更新，和DQN、DDPG这种膈几步就更新一次的不一样
        '''
        for _ in range(self.epochs):
            state, action, reward, done, old_prob, value, batches = self.memory.sample(batch_size=self.batch_size)
            # GAE
            advantage = np.zeros(len(reward), dtype=np.float32)
            for i in range(len(reward) - 1):
                discount = 1
                a_i = 0
                for k in range(i, len(reward) - 1):
                    # 这里要注意，当done的时候，gamma * v（next_state） == 0
                    a_i += discount * (reward[k] + self.gamma * value[k+1] * (1 - int(done[k])) - value[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[i] = a_i
            advantage = torch.tensor(advantage).to(device)
            value = torch.tensor(value).squeeze().to(device)
            for batch in batches:
                ## tmp_state = np.array([list(state[i].values()) for i in range(len(state))])
                batch_state = torch.tensor(np.array([list(state[i][0].values()) for i in range(len(state))])[batch], dtype=torch.float).to(device)
                batch_old_prob = torch.tensor(np.array(old_prob)[batch]).to(device)
                batch_action = torch.tensor(np.array([list(action[i][0].values()) for i in range(len(action))])[batch]).to(device)
                # 计算batch_new_prob
                bacth_logit = self.actor(batch_state)
                bacth_dist = Normal(bacth_logit['mu'], bacth_logit['sigma'])
                bacth_dist = Independent(bacth_dist, 1)
                batch_new_prob = bacth_dist.log_prob(batch_action)
                # 计算actor的loss，即核心的更新公式
                prob_ratio = batch_new_prob.exp() / batch_old_prob.exp()
                weighted_prob = prob_ratio * advantage[batch]
                weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage[batch]
                actor_loss = -torch.min(weighted_prob, weighted_clipped_prob).mean()
                # 更新actor网络
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # 计算critic的loss
                batch_critic_value = torch.squeeze(self.critic(batch_state))
                batch_real_value = advantage[batch] + value[batch]
                critic_loss = F.mse_loss(batch_real_value,
                                         batch_critic_value)  # critic_loss = (batch_real_value - batch_critic_value) ** 2    # critic_loss = critic_loss.mean()
                # 更新critic网络
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        self.memory.clear()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
