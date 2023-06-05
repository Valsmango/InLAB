# coding=utf-8
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from finalexperiment.chp3_static.env.env import Env
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
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
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l4)
        # orthogonal_init(self.l5)
        # orthogonal_init(self.l6)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class ReplayBuffer(object):
    def __init__(self, buffer_capacity, state_dim, action_dim):
        self.max_size = buffer_capacity
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 64  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.adaptive_alpha = False  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = ?dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.35

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.cpu().data.numpy().flatten()

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - float(self.alpha) * log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
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
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
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
        self.critic.load_state_dict(torch.load(filename + "_critic",map_location={'cuda:1':'cuda:0'}))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer",map_location={'cuda:1':'cuda:0'}))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor",map_location={'cuda:1':'cuda:0'}))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location={'cuda:1':'cuda:0'}))


if __name__ == '__main__':
    start_time = time.time()
    # env_name = "StaticEnv"
    env_name = "DynamicEnv"
    model_name = "SAC"
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

    max_train_steps = 3e6  # Maximum number of training steps
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration

    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(buffer_capacity=int(1e6), state_dim=state_dim, action_dim=action_dim)

    if not os.path.exists(f"./reward_train/{model_name}/{env_name}"):
        os.makedirs(f"./reward_train/{model_name}/{env_name}")
    if not os.path.exists(f"./model_train/{model_name}/{env_name}"):
        os.makedirs(f"./model_train/{model_name}/{env_name}")

    s = env.reset()
    done = False

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
            a = agent.choose_action(s)

        s_, r, done, category = env.step(a)
        if done and episode_timesteps < env._max_episode_steps:
            dw = True
        else:
            dw = False
        replay_buffer.store(s, a, r, s_, dw)  # Store the transition
        s = s_
        episode_reward += r

        if t >= random_steps:
            agent.learn(replay_buffer)

        if done:
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            train_episode_rewards.append(episode_reward)
            time_records.append(t)
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
                np.save(f'./reward_train/{model_name}/{env_name}/reward_seed_{seed}.npy',
                        np.array(train_episode_rewards))
                np.save(f'./reward_train/{model_name}/{env_name}/success_seed_{seed}.npy',
                        np.array(train_episode_success_rate))
                np.save(f'./reward_train/{model_name}/{env_name}/collision_seed_{seed}.npy',
                        np.array(train_episode_collision_rate))

            # Reset environment
            episode_num += 1
            env.close()
            env = Env()
            s, done = env.reset(), False
            episode_reward = 0.
            episode_timesteps = 0

    env.close()
    agent.save(f"./model_train/{model_name}/{env_name}/{model_name}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}")