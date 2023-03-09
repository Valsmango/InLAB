# coding=utf-8
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from graduation.envs.StandardEnv import *
import os
from typing import Dict, List, Tuple
import collections
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.hidden_width = hidden_width
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.LSTM(hidden_width, hidden_width, batch_first=True)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)

    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_width]), torch.zeros([1, batch_size, self.hidden_width])
        else:
            return torch.zeros([1, 1, self.hidden_width]), torch.zeros([1, 1, self.hidden_width])

    def atten(self, lstm_output, h_t):
        '''
        参考的代码为：
        k = h
        q = lstm_output
        v = h
        lstm_output为 batch，seq_len，hidden_width【1，len，256】   h为 1，batch，hidden_width【1，1，256】
        （1）先进行规格的转换，变为：lstm_output规格为 batch，len, dim； ht规格为 batch, dim，1
            h_t = h_t.permute(1, 2, 0)
        （2）计算weights、attention，规格都为 batch，len，1
            attn_weights = torch.bmm(lstm_output, h_t)
            attention = F.softmax(attn_weights, 1)
        （3）进行规格的转换，变为：lstm_output规格为 batch，dim, len
            attn_out = torch.bmm(lstm_output.transpose(1, 2), attention)
        （4）输出即为batch, dim，1，最终规格改回1，batch, dim
            attn_out = attn_out.permute(2, 0, 1)

            self-atten:  softmax( qk' / sqrt(in_dim)) * v
        ____________________________________________________________________
        k = lstm_output
        q = h
        v = lstm_output
        lstm_output为 batch，seq_len，hidden_width【1，len，256】   h为 1，batch，hidden_width【1，1，256】
        （1）先进行规格的转换，变为：q = ht规格为 batch, len, dim
            q = h_t.permute(1, 0, 2)  # 【batch，1，dim】
            trans = torch.ones(h_t.shape[1], lstm_output.shape[1], h_t.shape[0]).to(device)  # 【batch，len，1】
            q = torch.bmm(trans, q)  # 【batch，len, dim】

        （2）转置 k = lstm_output规格为 batch，dim, len； 计算weights、attention，规格都为 batch，len, len
            k = lstm_output.permute(0, 2, 1)
            attn_weights = torch.bmm(q, k)
            attention = F.softmax(attn_weights, 1)
        （3）进行规格的转换，变为：v = lstm_output规格为 batch，len, dim    而attention的规格应该是 batch，len，len
            v = lstm_output
            attn_out = torch.bmm(attention, v)
        （4）输出即为batch, len，dim
        '''
        # q = h_t.permute(1, 0, 2)  # 【batch，1，dim】
        # trans = torch.ones(h_t.shape[1], lstm_output.shape[1], h_t.shape[0]).to(device)  # 【batch，len，1】
        # q = torch.bmm(trans, q)  # 【batch，len, dim】
        # k = lstm_output.permute(0, 2, 1)
        # attn_weights = torch.bmm(q, k)
        # attention = F.softmax(attn_weights, 1)
        # v = lstm_output
        # attn_out = torch.bmm(attention, v)
        # return attn_out
        h_t = h_t.permute(1, 2, 0)
        attn_weights = torch.bmm(lstm_output, h_t)
        attention = F.softmax(attn_weights, 1)
        attn_out = torch.bmm(lstm_output.transpose(1, 2), attention)
        attn_out = attn_out.permute(2, 0, 1)
        return attn_out

    def forward(self, x, h_0, c_0, deterministic=False, with_logprob=True):
        if not hasattr(self, '_flattened'):
            self.l3.flatten_parameters()
            setattr(self, '_flattened', True)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x, (h_t, c_t) = self.l3(x, (h_0, c_0))
        h_t = self.atten(x, h_t)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=2, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=2, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi, h_t, c_t


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

    def init_hidden_state(self, batch_size, training=None):
        '''
        https://www.cnblogs.com/jiangkejie/p/13246857.html
        '''
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_width]), torch.zeros([1, batch_size, self.hidden_width])
        else:
            return torch.zeros([1, 1, self.hidden_width]), torch.zeros([1, 1, self.hidden_width])

    def atten(self, lstm_output, h_t):
        h_t = h_t.permute(1, 2, 0)
        attn_weights = torch.bmm(lstm_output, h_t)
        attention = F.softmax(attn_weights, 1)
        attn_out = torch.bmm(lstm_output.transpose(1, 2), attention)
        attn_out = attn_out.permute(2, 0, 1)
        return attn_out

    def self_atten(self, x):
        pass

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
        h_t_1 = self.atten(q1, h_t_1)
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(s_a))
        q2 = F.relu(self.l6(q2))
        q2, (h_t_2, c_t_2) = self.l7(q2, (h_0_2, c_0_2))   # q2 = F.relu(self.l4(s_a))
        h_t_2 = self.atten(q2, h_t_2)
        q2 = self.l8(q2)

        return q1, q2, h_t_1, c_t_1, h_t_2, c_t_2


class SACLSTM(object):
    def __init__(self, batch_size, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = batch_size  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = ?dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, h, c, deterministic=False):
        s = torch.FloatTensor(s.reshape(1, -1)).unsqueeze(0).to(device)
        h = torch.FloatTensor(h).to(device)
        c = torch.FloatTensor(c).to(device)
        a, _, h, c = self.actor(s, h, c, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.cpu().data.numpy().flatten(), h.cpu().data.numpy(), c.cpu().data.numpy()

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch
        # batch, seq_len, dim
        h_target_1, c_target_1 = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_target_2, c_target_2 = copy.deepcopy(h_target_1), copy.deepcopy(c_target_1)
        h_target_1, c_target_1 = h_target_1.to(device), c_target_1.to(device)
        h_target_2, c_target_2 = h_target_2.to(device), c_target_2.to(device)
        h_, c_ = self.actor.init_hidden_state(batch_size=batch_size, training=True)
        h_, c_ = h_.to(device), c_.to(device)
        with torch.no_grad():
            batch_a_, log_pi_, h_, c_ = self.actor(batch_s_, h_, c_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2, h_target_1, c_target_1, h_target_2, c_target_2 = \
                self.critic_target(batch_s_, batch_a_, h_target_1, c_target_1, h_target_2, c_target_2 )
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - float(self.alpha) * log_pi_)

        # Compute current Q
        h_current, c_current = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current, c_current = h_current.to(device), c_current.to(device)
        h_current_, c_current_ = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current_, c_current_ = h_current_.to(device), c_current_.to(device)
        current_Q1, current_Q2, h_current, c_current, h_current_, c_current_= self.critic(batch_s, batch_a, h_current, c_current, h_current_, c_current_)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        h, c = self.actor.init_hidden_state(batch_size=batch_size, training=True)
        h, c = h.to(device), c.to(device)
        a, log_pi, h, c = self.actor(batch_s, h, c)
        h_1, c_1 = self.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_2, c_2 = copy.deepcopy(h_1), copy.deepcopy(c_1)
        h_1, c_1 = h_1.to(device), c_1.to(device)
        h_2, c_2 = h_2.to(device), c_2.to(device)
        Q1, Q2, h_1, c_1, h_2, c_2 = self.critic(batch_s, a, h_1, c_1, h_2, c_2)
        Q = torch.min(Q1, Q2)
        actor_loss = (float(self.alpha) * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp().to(device) * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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


def evaluate_policy(agent, batch_size):
    times = 40  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        h, c = agent.actor.init_hidden_state(batch_size=batch_size, training=False)
        env = StandardEnv()
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a, h, c = agent.choose_action(s, h, c, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done = env.step(a)
            episode_reward += r
            s = s_
        env.close()
        evaluate_reward += episode_reward

    return evaluate_reward / times


if __name__ == '__main__':
    start_time = time.time()
    env_name = "StandardEnv"
    env = StandardEnv()
    # Set random seed
    seed = 10
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = 12
    action_dim = 3
    max_action = 1.0
    max_episode_steps = 200  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    max_train_steps = 2e6  # Maximum number of training steps
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    batch_size = 1
    lookup_step = 16

    agent = SACLSTM(batch_size, state_dim, action_dim, max_action)
    replay_buffer = EpisodeMemory(random_update=False,
                                   max_epi_num=20000, max_epi_len=200,
                                   batch_size=batch_size,
                                   lookup_step=lookup_step)    # Build a tensorboard

    # writer = SummaryWriter(log_dir='runs/SAC/SAC_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))

    if not os.path.exists("./eval_reward_train"):
        os.makedirs("./eval_reward_train")
    if not os.path.exists("./model_train/SACLSTMATT"):
        os.makedirs("./model_train/SACLSTMATT")

    while total_steps < max_train_steps:
        env = StandardEnv()
        s = env.reset()
        done = False
        episode_steps = 0
        episode_record = EpisodeBuffer()
        h, c = agent.actor.init_hidden_state(batch_size=batch_size, training=False)
        while not done:
            episode_steps += 1
            if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.sample_action()
            else:
                a, h, c = agent.choose_action(s, h, c)
            s_, r, done = env.step(a)
            # r = reward_adapter(r, env_index)  # Adjust rewards for better performance
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False
            episode_record.put([s, a, r, s_, dw])
            s = s_

            if total_steps >= random_steps:
                agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(agent, batch_size)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                # writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./eval_reward_train/SACLSTMATT_env_{}_seed_{}.npy'.format(env_name, seed), np.array(evaluate_rewards))

            total_steps += 1
        replay_buffer.put(episode_record)
        env.close()
    agent.save(f"./model_train/SACLSTM/SACLSTMATT_{env_name}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}")