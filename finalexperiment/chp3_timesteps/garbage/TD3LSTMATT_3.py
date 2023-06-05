# coding=utf-8
import gym
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from finalexperiment.chp3_timesteps.env.env import Env
from typing import Dict, List, Tuple
import collections
import os
import time
import math
from tqdm import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.hidden_width = hidden_width
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.LSTM(hidden_width, hidden_width, batch_first=True)
        self.l4 = nn.Linear(hidden_width, action_dim)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3, gain=0.01)

    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_width]), torch.zeros([1, batch_size, self.hidden_width])
        else:
            return torch.zeros([1, 1, self.hidden_width]), torch.zeros([1, 1, self.hidden_width])

    def atten(self, lstm_output, h_t):
        # lstm_output 【batch, seq_len, hidden==dim】
        # h_t  【1, batch, dim】
        q = h_t.permute(1, 0, 2)  # 【batch，1，dim】
        trans = torch.ones(h_t.shape[1], lstm_output.shape[1], h_t.shape[0]).to(device)  # 【batch，len，1】
        q = torch.bmm(trans, q)  # 【batch，len, dim】
        k = lstm_output.permute(0, 2, 1)

        # attn_weights = torch.bmm(q, k)
        d_k = k.size(-1)
        attn_weights = torch.bmm(q, k) / math.sqrt(d_k)

        attention = F.softmax(attn_weights, 1)
        v = lstm_output
        attn_out = torch.bmm(attention, v)
        return attn_out

    def forward(self, s, h_0, c_0):
        # https://blog.csdn.net/feifei3211/article/details/102998288?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-102998288-blog-109586782.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-102998288-blog-109586782.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=1
        if not hasattr(self, '_flattened'):
            self.l3.flatten_parameters()
            setattr(self, '_flattened', True)
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        s, (h_t, c_t) = self.l3(s, (h_0, c_0))
        a = self.max_action * torch.tanh(self.l4(s))  # [-max,max]
        return a, h_t, c_t


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.hidden_width = hidden_width
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.LSTM(hidden_width, hidden_width, batch_first=True)
        self.l4 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3)
        # Q2
        self.l5 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l6 = nn.Linear(hidden_width, hidden_width)
        self.l7 = nn.LSTM(hidden_width, hidden_width, batch_first=True)
        self.l8 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l4)
        # orthogonal_init(self.l5)
        # orthogonal_init(self.l6)

        # self.w_q = nn.Linear(hidden_width, hidden_width, bias=False)
        # self.w_k = nn.Linear(hidden_width, hidden_width, bias=False)
        # self.w_v = nn.Linear(hidden_width, hidden_width, bias=False)

    def init_hidden_state(self, batch_size, training=None):
        '''
        https://www.cnblogs.com/jiangkejie/p/13246857.html
        '''
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_width]), torch.zeros([1, batch_size, self.hidden_width])
        else:
            return torch.zeros([1, 1, self.hidden_width]), torch.zeros([1, 1, self.hidden_width])

    # def atten(self, lstm_output):
    #     # lstm_output 【batch, seq_len, hidden】
    #     q = self.w_q(lstm_output)
    #     k = self.w_k(lstm_output)
    #     v = self.w_v(lstm_output)    # 【batch, seq_len, hidden】
    #
    #     d_k = k.size(-1)
    #     scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k)    # 【batch, seq_len, seq_len】
    #     alpha_n = F.softmax(scores, dim=-1)
    #     attn_out = torch.matmul(alpha_n, v)
    #     # attn_out = attn_out.sum(1)
    #     return attn_out
    def atten(self, lstm_output, h_t):
        q = h_t.permute(1, 0, 2)  # 【batch，1，dim】
        trans = torch.ones(h_t.shape[1], lstm_output.shape[1], h_t.shape[0]).to(device)  # 【batch，len，1】
        q = torch.bmm(trans, q)  # 【batch，len, dim】
        k = lstm_output.permute(0, 2, 1)

        # attn_weights = torch.bmm(q, k)
        d_k = k.size(-1)
        attn_weights = torch.bmm(q, k) / math.sqrt(d_k)

        attention = F.softmax(attn_weights, 1)
        v = lstm_output
        attn_out = torch.bmm(attention, v)
        return attn_out

    def forward(self, s, a, h_0_1, c_0_1, h_0_2, c_0_2):
        if not hasattr(self, '_flattened'):
            self.l3.flatten_parameters()
            self.l7.flatten_parameters()
            setattr(self, '_flattened', True)
        # s_a = torch.cat([s, a], 1)
        s_a = torch.cat([s, a], 2)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1, (h_t_1, c_t_1) = self.l3(q1, (h_0_1, c_0_1))  # q1 = F.relu(self.l1(s_a))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(s_a))
        q2 = F.relu(self.l6(q2))
        q2, (h_t_2, c_t_2) = self.l7(q2, (h_0_2, c_0_2))   # q2 = F.relu(self.l4(s_a))

        q2 = self.l8(q2)

        return q1, q2, h_t_1, c_t_1, h_t_2, c_t_2

    def Q1(self, s, a, h_0_1, c_0_1):
        if not hasattr(self, '_flattened'):
            self.l3.flatten_parameters()
            setattr(self, '_flattened', True)
        # s_a = torch.cat([s, a], 1)
        s_a = torch.cat([s, a], 2)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1, (h_t_1, c_t_1) = self.l3(q1, (h_0_1, c_0_1))  # q1 = F.relu(self.l1(s_a))

        q1 = self.l4(q1)

        return q1, h_t_1, c_t_1


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                 max_epi_num=20000, max_epi_len=200,
                 batch_size=1,
                 lookup_step=None):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit(
                'It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def _pre_sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)

            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def sample(self, batch_size):
        samples, seq_len = self._pre_sample()
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

        observations = torch.FloatTensor(observations).reshape(batch_size, seq_len, -1).to(device)
        actions = torch.FloatTensor(actions).reshape(batch_size, seq_len, -1).to(device)
        rewards = torch.FloatTensor(rewards).reshape(batch_size, seq_len, -1).to(device)
        next_observations = torch.FloatTensor(next_observations).reshape(batch_size, seq_len, -1).to(device)
        dones = torch.FloatTensor(dones).reshape(batch_size, seq_len, -1).to(device)
        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(np.array(transition[0]))
        self.action.append(np.array(transition[1]))
        self.reward.append(np.array(transition[2]))
        self.next_obs.append(np.array(transition[3]))
        self.done.append(np.array(transition[4]))

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



class TD3LSTMATT(object):
    def __init__(self, batch_size, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = batch_size  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.policy_noise = 0.2 * max_action  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5 * max_action  # Clip the noise
        self.policy_freq = 2  # The frequency of policy updates
        self.actor_pointer = 0

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, h, c):
        # s = torch.unsqueeze(s, 0)   # s = torch.FloatTensor(s.reshape(1, -1))
        s = torch.FloatTensor(s.reshape(1, -1)).unsqueeze(0).to(device)
        h = torch.FloatTensor(h).to(device)
        c = torch.FloatTensor(c).to(device)
        a, h, c = self.actor(s, h,  c)
        return a.cpu().data.numpy().flatten(), h.cpu().data.numpy(), c.cpu().data.numpy()

    def learn(self, relay_buffer):
        self.actor_pointer += 1
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            h_target_0, c_target_0 = self.actor.init_hidden_state(batch_size=self.batch_size, training=True)
            h_target_0, c_target_0 = h_target_0.to(device), c_target_0.to(device)

            h_target_1, c_target_1 = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
            h_target_2, c_target_2 = copy.deepcopy(h_target_1), copy.deepcopy(c_target_1)
            h_target_1, c_target_1 = h_target_1.to(device), c_target_1.to(device)
            h_target_2, c_target_2 = h_target_2.to(device), c_target_2.to(device)
            # Trick 1:target policy smoothing
            # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, h_target_0, c_target_0 = self.actor_target(batch_s_, h_target_0, c_target_0)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            # Trick 2:clipped double Q-learning
            target_Q1, target_Q2, h_target_1, c_target_1, h_target_2, c_target_2 = \
                self.critic_target(batch_s_, next_action, h_target_1, c_target_1, h_target_2, c_target_2)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)

        # Get the current Q
        h_current, c_current = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current, c_current = h_current.to(device), c_current.to(device)
        h_current_, c_current_ = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current_, c_current_ = h_current_.to(device), c_current_.to(device)
        current_Q1, current_Q2, h_current, c_current, h_current_, c_current_ = \
            self.critic(batch_s, batch_a, h_current,c_current, h_current_, c_current_)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_freq == 0:
            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            # Compute actor loss
            h, c = self.actor.init_hidden_state(batch_size=batch_size, training=True)
            h, c = h.to(device), c.to(device)
            a, h, c = self.actor(batch_s, h, c)
            h_1, c_1 = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
            h_1, c_1 = h_1.to(device), c_1.to(device)
            actor_loss, h_1, c_1 = self.critic.Q1(batch_s, a, h_1, c_1)  # Only use Q1
            actor_loss = -actor_loss.mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic networks
            for params in self.critic.parameters():
                params.requires_grad = True

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


if __name__ == '__main__':
    start_time = time.time()
    # env_name ="StaticEnv"
    env_name = "DynamicEnv"
    model_name = "TD3_LSTM_ATT_3"
    env = Env()
    # Set random seed
    seed = 10
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_dim = 12
    action_dim = 3
    max_action = 1.0
    max_episode_steps = 200  # Maximum number of steps per episode

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 2e6  # Maximum number of training steps
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration

    batch_size = 64 # random --> 64, sequential --> 1
    lookup_step = 16
    agent = TD3LSTMATT(batch_size, state_dim, action_dim, max_action)
    replay_buffer = EpisodeMemory(random_update=True,
                                  max_epi_num=20000, max_epi_len=200,
                                   batch_size=batch_size,
                                   lookup_step=lookup_step)

    if not os.path.exists(f"./reward_train/{model_name}/{env_name}"):
        os.makedirs(f"./reward_train/{model_name}/{env_name}")
    if not os.path.exists(f"./model_train/{model_name}/{env_name}"):
        os.makedirs(f"./model_train/{model_name}/{env_name}")

    s = env.reset()
    done = False
    episode_record = EpisodeBuffer()
    h, c = agent.actor.init_hidden_state(batch_size=batch_size, training=False)

    train_episode_rewards = []
    episode_reward = 0.
    episode_timesteps = 0
    episode_num = 0
    train_episode_success_rate = []
    train_episode_collision_rate = []
    time_records = []

    for t in tqdm(range(int(max_train_steps))):
        episode_timesteps += 1

        if t < random_steps:  # Take the random actions in the beginning for the better exploration
            a = env.sample_action()
        else:
            a, h, c = agent.choose_action(s, h, c)
            a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)

        s_, r, done, category = env.step(a)

        if done and episode_timesteps < env._max_episode_steps:
            dw = True
        else:
            dw = False
        episode_record.put([s, a, r, s_, dw])
        s = s_
        episode_reward += r

        if t >= random_steps:
            agent.learn(replay_buffer)

        if done:
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            time_records.append(t)
            train_episode_rewards.append(episode_reward)
            if category == 1:
                success_rate = 1
                collision_rate = 0
            elif category == 2:
                success_rate = 0
                collision_rate = 1
            else:
                success_rate = 0
                collision_rate = 0
            train_episode_success_rate.append(success_rate)
            train_episode_collision_rate.append(collision_rate)

            if (episode_num + 1) % 100 == 0:
                np.save(f'./reward_train/{model_name}/{env_name}/timestep_seed_{seed}.npy', np.array(time_records))
                np.save(f'./reward_train/{model_name}/{env_name}/reward_seed_{seed}.npy', np.array(train_episode_rewards))
                np.save(f'./reward_train/{model_name}/{env_name}/success_seed_{seed}.npy', np.array(train_episode_success_rate))
                np.save(f'./reward_train/{model_name}/{env_name}/collision_seed_{seed}.npy', np.array(train_episode_collision_rate))


            # Reset environment
            replay_buffer.put(episode_record)
            episode_num += 1
            env.close()
            env = Env()
            s, done = env.reset(), False
            episode_record = EpisodeBuffer()
            h, c = agent.actor.init_hidden_state(batch_size=batch_size, training=False)
            episode_timesteps = 0
            episode_reward = 0.


    env.close()
    agent.save(f"./model_train/{model_name}/{env_name}/{model_name}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}")