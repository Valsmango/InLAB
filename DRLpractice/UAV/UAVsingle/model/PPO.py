# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter

# set device
from DRLpractice.UAV.UAVsingle.replaybuffer import PPODiscreteReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x = F.softmax(self.Linear3(x), dim=-1)  # softmax 从最后一个维度上进行计算，得到的数据维度还是[batch_size或者1, 2]， 2为action_space，即每个action的被选择概率
        x = Categorical(x)
        return x


class DiscretePPO:
    def __init__(self, batch_size, state_space, action_space, clip,
                 actor_lr, critic_lr, epochs, gamma, gae_lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.clip = clip

        self.batch_size = batch_size
        self.memory = PPODiscreteReplayBuffer(self.batch_size)
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
        action = torch.squeeze(action).item()   # 因为action还是size（[batch_size或1 , 2]）维度的数据，实则只需要size（[2]）的数据

        value = self.critic(state)
        value = torch.squeeze(value).item()  # 因为value还是size（[batch_size或1 , 1]）维度的数据，实则只需要size（[1]）的数据

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
                    # 这里要注意，当done的时候，gamma * v（next_state） == 0
                    a_i += discount * (reward[k] + self.gamma * value[k+1] * int(done[k]) - value[k])
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
                batch_new_prob = distribution.log_prob(batch_action)
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