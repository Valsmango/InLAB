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
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import time


class CriticNetwork(nn.Module):
    def __init__(self, output_size):
        # 继承nn.Module：
        super(CriticNetwork, self).__init__()
        '''
        待填充，网络结构
        '''

    def forward(self, x):
        # x 即为 state
        '''
        待填充，数据格式转换
        '''
        return self.linear(x)


class ActorNetwork(nn.Module):
    def __init__(self, output_size):
        # 继承nn.Module：
        super(CriticNetwork, self).__init__()
        '''
        待填充，网络结构
        '''

    def forward(self, x):
        # x 即为 state
        '''
        待填充，数据格式转换
        '''
        return self.linear(x)

    def sample_action(self, state, epsilon):
        output = self.forward(state)

        if random.random() < epsilon:
            return random.randint(0, 1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1], output[2]


class ReplayBuffer:
    def __init__(self, max_epi_num=100, max_epi_len=500, batch_size=1):
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.replaybuffer = collections.deque(maxlen=self.max_epi_num)

    def put(self, subreplaybuffer):
        self.replaybuffer.append(subreplaybuffer)  # 每一个subreplaybuffer == 一个episode

    def sample(self):
        sampled_buffer = []
        '''
        待填充，选出samples
        '''

        return sampled_buffer  # buffers, sequence_length

    def __len__(self):
        return len(self.replaybuffer)


class SubReplayBuffer:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, s, a, r, next_s, done):
        self.obs.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.next_obs.append(next_s)
        self.done.append(done)

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx + lookup_step]
            action = action[idx:idx + lookup_step]
            reward = reward[idx:idx + lookup_step]
            next_obs = next_obs[idx:idx + lookup_step]
            done = done[idx:idx + lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


def optimize_model(Q_net=None, Q_target_net=None, replay_buffer=None, batch_size=1, gamma=0.99):
    samples = replay_buffer.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    '''
    待填充，讲数据转换为Tensor，并计算网络的Loss
    '''


    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    t0 = time.time()

    # Env parameters
    model_name = "PPO"
    env_name = "CartPole-v0"
    seed = 10
    save_model_flag = True

    # Set gym environment
    env = gym.make(env_name)

    # Set seeds
    env.seed(seed)
    # env.action_space.seed(seed)     # 这个是TD3中提到的seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save results
    file_name = f"{model_name}_{env_name}_{seed}"
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model_flag and not os.path.exists("./models"):
        os.makedirs("./models")

    # model parameters
    # 适当设置batch size，过小的经验池容量和batchsize导致收敛到局部最优，结果呈现震荡形式
    learning_rate = 1e-3
    batch_size = 16
    tau = 1e-2
    max_steps = 2000
    max_episodes = 600
    min_epi_num = 64  # Start moment to train the Q network

    # Initiate the network and set the optimizer
    env.reset()
    n_actions = env.action_space.n
    Q_net = CriticNetwork(n_actions).to(device)
    Q_target_net = CriticNetwork(n_actions).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=learning_rate)

    # Initiate the ReplayBuffer
    replay_buffer = ReplayBuffer(max_epi_num=200, max_epi_len=600,
                                 random_update=True, batch_size=batch_size, lookup_step=20)

    # other parameters
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    epsilon = eps_start
    gamma = 0.99
    TARGET_UPDATE = 4

    # output the reward
    print_per_iter = 20
    score = 0
    score_sum = 0.0

    # 开始训练
    for i in range(max_episodes):
        # Initialize the environment and state
        state = env.reset()
        h, c = Q_net.init_hidden_state(batch_size=batch_size, training=False)

        # 创建存储当前episode的临时buffer
        sub_replay_buffer = SubReplayBuffer()

        for t in range(max_steps):
            # Select and perform an action
            action, h, c = Q_net.sample_action(state.unsqueeze(0).unsqueeze(0).to(device),
                                               # state 为 B seq CHW     B x seq x 3 x 160 x 360
                                               h=h.to(device),
                                               c=c.to(device),
                                               epsilon=epsilon)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state[::2]
            # reward是一个float格式的数值
            score += reward
            score_sum += reward
            done_mask = 0.0 if done else 1.0
            sub_replay_buffer.put(s=state, a=action, next_s=next_state, r=reward, done=done_mask)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if len(replay_buffer) >= min_epi_num:
                optimize_model(Q_net=Q_net, Q_target_net=Q_target_net, replay_buffer=replay_buffer,
                               batch_size=batch_size, gamma=gamma)

                if (t + 1) % TARGET_UPDATE == 0:
                    # Q_target_net.load_state_dict(Q_net.state_dict()) # naive update
                    for target_param, local_param in zip(Q_target_net.parameters(),
                                                         Q_net.parameters()):  # <- soft update
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing
        replay_buffer.put(sub_replay_buffer)
        score = 0

        # 隔一段时间，输出一次训练的结果
        if i % print_per_iter == 0 and i != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                i, score_sum / print_per_iter, len(replay_buffer), epsilon * 100))
            score_sum = 0.0

    # save the model
    if save_model_flag:
        torch.save(Q_net.state_dict(), f"./models/{file_name}")

    t1 = time.time()
    print(f"运行时间为：{t1 - t0:.2f}s")
