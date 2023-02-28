# coding=utf-8
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
from graduation.envs.StandardEnv import *
import numpy as np
import os
import time
import matplotlib as plt

'''
最原始版本的PPO，和另一个加了很多trick的不同
'''

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PPOReplayBuffer(object):
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []

    def add(self, s, a, r, done, p, v):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(done)
        self.prob.append(p)
        self.value.append(v)

    def clear(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.value = []

    def sample(self, batch_size):
        batch_step = np.arange(0, len(self.done),
                               batch_size)  # np.arange(start = 0, stop = len, step = batch_sz) 即： 【0， batch_sz, batch_sz*2, ... < stop】
        indicies = np.arange(len(self.done), dtype=np.int64)  # np.arange(start = 0, stop = len, step = 1)
        np.random.shuffle(indicies)  # 将indicies弄成乱序
        batches = [indicies[i:i + batch_size] for i in batch_step]
        return self.state, self.action, self.reward, self.done, self.prob, self.value, batches

    def __len__(self) -> int:
        return len(self.done)


class CriticNetwork(nn.Module):
    '''
    Critic输入为：state
    输出为： value(state)的估计值
    '''

    def __init__(self, state_space, output_size=1):
        super(CriticNetwork, self).__init__()

        self.hidden_space = 256
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

        self.hidden_space = 256
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)  # linear3输出均值mu
        self.max_action = max_action
        self.log_sigma = nn.Parameter(torch.zeros(1, action_space))

        orthogonal_init(self.Linear1)
        orthogonal_init(self.Linear2)
        orthogonal_init(self.Linear3, gain=0.01)

    def forward(self, x):
        x = torch.tanh(self.Linear1(x))  # 有的用的tanh进行激活
        x = torch.tanh(self.Linear2(x))
        mean = self.max_action * torch.tanh(self.Linear3(x))  # 连续动作空间不需要softmax，而是要改进成输出一个均值mu
        log_std = self.log_sigma.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class PPO:
    def __init__(self, batch_size, state_space, action_space, clip,
                 actor_lr, critic_lr, epochs, gamma, gae_lambda, max_action):
        self.state_space = state_space
        self.action_space = action_space
        self.clip = clip

        self.batch_size = batch_size
        self.memory = PPOReplayBuffer()
        self.actor = ActorNetwork(state_space=self.state_space, action_space=self.action_space,
                                  max_action=max_action).to(device)
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
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        dist = self.actor(state)
        # 为一个 batch 里的每个样本采样一个维度为 ``action_shape`` 的连续动作，并返回它
        action = dist.sample()
        probs = dist.log_prob(action)
        # action = torch.squeeze(action).item()  # 因为action还是size（[batch_size或1 , 2]）维度的数据，实则只需要size（[2]）的数据
        # probs = torch.squeeze(probs).item()

        value = self.critic(state)
        # # value = torch.squeeze(value).item()  # 因为value还是size（[batch_size或1 , 1]）维度的数据，实则只需要size（[1]）的数据

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
                    a_i += discount * (reward[k] + self.gamma * value[k + 1] * (1 - int(done[k])) - value[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[i] = a_i
            advantage = torch.tensor(advantage).to(device)
            value = torch.tensor(value).squeeze().to(device)
            for batch in batches:
                batch_state = torch.tensor(state[batch]).to(device)
                batch_old_prob = torch.tensor(old_prob[batch]).to(device)
                batch_action = torch.tensor(action[batch]).to(device)
                # 计算batch_new_prob
                bacth_dist = self.actor(batch_state)
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


def evaluate_policy(agent):
    times = 40
    evaluate_reward = 0
    for _ in range(times):
        env = StandardEnv()
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a, _ = agent.select_action(s)  # We use the deterministic policy during the evaluating
            a = np.clip(a, -1, 1)
            s_, r, done = env.step(a)
            episode_reward += r
            s = s_
        env.close()
        evaluate_reward += episode_reward

    return evaluate_reward / times


if __name__ == "__main__":
    starttime = time.time()
    eval_records = []

    # Env parameters
    model_name = "PPO2"
    env_name = "StandardEnv"
    seed = 10
    # writer = SummaryWriter("./runs/tensorboard/ppo")
    env = StandardEnv()
    # Set random seed
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = 12
    action_dim = 3
    max_action = 1.0
    max_episode_steps = 200

    # save results
    if not os.path.exists("./eval_reward_train"):
        os.makedirs("./eval_reward_train")
    if not os.path.exists("./model_train/PPO2"):
        os.makedirs("./model_train/PPO2")

    print("---------------------------------------")
    print(f"Policy: {model_name}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    # model parameters
    # 适当设置batch size，过小的经验池容量和batchsize导致收敛到局部最优，结果呈现震荡形式
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-3
    batch_size = 16
    max_train_steps = int(1e6)
    evaluate_freq = 5e3
    clip = 0.2
    gamma = 0.99
    gae_lambda = 0.92
    step_count = 0
    epochs = 10
    update_frequent = 200  # 每走200步更新一次

    # Initiate the network and set the optimizer
    model = PPO(batch_size=batch_size, state_space=state_dim, action_space=action_dim, clip=clip,
                actor_lr=actor_learning_rate, critic_lr=critic_learning_rate,
                epochs=epochs, gamma=gamma, gae_lambda=gae_lambda, max_action=max_action)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    save_freq = 20

    # 开始训练
    # for i in tqdm(range(max_episodes)):
    while total_steps < max_train_steps:
        # Initialize the environment and state
        env = StandardEnv()
        state = env.reset()
        episode_steps = 0
        done = None
        score = 0
        while not done:
            episode_steps += 1
            # 渲染
            # env.render()
            # Select and perform an action
            action, prob, val = model.select_action(state)
            action = np.clip(action, -1, 1)
            # print(action)
            next_state, reward, done = env.step(action)
            # next_state = next_state[::2]
            # reward是一个float格式的数值
            score += reward
            done_bool = float(done)
            model.memory.add(s=state, a=action, r=reward, done=done_bool, p=prob, v=val)

            # Move to the next state
            state = next_state
            total_steps += 1

            # Perform one step of the optimization
            if total_steps % update_frequent == 0:
                model.update()
            if total_steps % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(model)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                if evaluate_num % save_freq == 0:
                    np.save('./eval_reward_train/PPO2_env_{}_seed_{}.npy'.format(env_name, seed), np.array(evaluate_rewards))

        env.close()
    # save the model
    model.save(f"./model_train/PPO/PPO_{env_name}")

    endtime = time.time()
    dtime = endtime - starttime
    print("-----------------------------------------")
    print("程序运行时间：%.8s s" % dtime)
    print("-----------------------------------------")
