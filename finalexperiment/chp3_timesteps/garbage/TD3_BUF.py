# coding=utf-8
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import *
import copy
from torch.utils.tensorboard import SummaryWriter
from finalexperiment.chp3_timesteps.env.env import Env
import os
import time

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
        self.l3 = nn.Linear(hidden_width, action_dim)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3, gain=0.01)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a


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

    def Q1(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class ReplayBuffer(object):
    def __init__(self, buffer_capacity, state_dim, action_dim):
        self.max_size = buffer_capacity
        self.count = 0
        self.size = 0
        # self.s = np.zeros((self.max_size, state_dim))
        # self.a = np.zeros((self.max_size, action_dim))
        # self.r = np.zeros((self.max_size, 1))
        # self.s_ = np.zeros((self.max_size, state_dim))
        # self.dw = np.zeros((self.max_size, 1))

        self.avg_r = 0.
        self.better_count = 0
        self.better_size = 0
        self.better_max_size = int(self.max_size/2)
        self.better_s = np.zeros((self.better_max_size, state_dim))
        self.better_a = np.zeros((self.better_max_size, action_dim))
        self.better_r = np.zeros((self.better_max_size, 1))
        self.better_s_ = np.zeros((self.better_max_size, state_dim))
        self.better_dw = np.zeros((self.better_max_size, 1))

        self.worse_count = 0
        self.worse_size = 0
        self.worse_max_size = self.max_size - self.better_max_size
        self.worse_s = np.zeros((self.worse_max_size, state_dim))
        self.worse_a = np.zeros((self.worse_max_size, action_dim))
        self.worse_r = np.zeros((self.worse_max_size, 1))
        self.worse_s_ = np.zeros((self.worse_max_size, state_dim))
        self.worse_dw = np.zeros((self.worse_max_size, 1))

    def store(self, s, a, r, s_, dw):
        # self.s[self.count] = s
        # self.a[self.count] = a
        # self.r[self.count] = r
        # self.s_[self.count] = s_
        # self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions
        if r > self.avg_r:
            self.better_s[self.better_count] = s
            self.better_a[self.better_count] = a
            self.better_r[self.better_count] = r
            self.better_s_[self.better_count] = s_
            self.better_dw[self.better_count] = dw
            self.better_count = (self.better_count + 1) % self.better_max_size
            self.better_size = min(self.better_size + 1, self.better_max_size)
        else:
            self.worse_s[self.worse_count] = s
            self.worse_a[self.worse_count] = a
            self.worse_r[self.worse_count] = r
            self.worse_s_[self.worse_count] = s_
            self.worse_dw[self.worse_count] = dw
            self.worse_count = (self.worse_count + 1) % self.worse_max_size
            self.worse_size = min(self.worse_size + 1, self.worse_max_size)
        self.avg_r = self.avg_r + (r - self.avg_r)/self.count


    def sample(self, batch_size):
        better_num = int(batch_size * 0.5)
        worse_num = batch_size - better_num
        better_index = np.random.choice(self.better_size, size=better_num)
        worse_index = np.random.choice(self.worse_size, size=worse_num)
        batch_s = []
        batch_a = []
        batch_r = []
        batch_s_ = []
        batch_dw = []
        batch_s.extend(self.better_s[better_index])
        batch_s.extend(self.worse_s[worse_index])
        batch_s = torch.FloatTensor(batch_s).to(device)
        batch_a.extend(self.better_a[better_index])
        batch_a.extend(self.worse_a[worse_index])
        batch_a = torch.FloatTensor(batch_a).to(device)
        batch_r.extend(self.better_r[better_index])
        batch_r.extend(self.worse_r[worse_index])
        batch_r = torch.FloatTensor(batch_r).to(device)
        batch_s_.extend(self.better_s_[better_index])
        batch_s_.extend(self.worse_s_[worse_index])
        batch_s_ = torch.FloatTensor(batch_s_).to(device)
        batch_dw.extend(self.better_dw[better_index])
        batch_dw.extend(self.worse_dw[worse_index])
        batch_dw = torch.FloatTensor(batch_dw).to(device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 64  # batch size
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

    def choose_action(self, s):
        # s = torch.unsqueeze(s, 0)   # s = torch.FloatTensor(s.reshape(1, -1))
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        a = self.actor(s).cpu().data.numpy().flatten()
        return a

    def learn(self, relay_buffer):
        self.actor_pointer += 1
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(batch_s_) + noise).clamp(-self.max_action, self.max_action)
            # Trick 2:clipped double Q-learning
            target_Q1, target_Q2 = self.critic_target(batch_s_, next_action)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)

        # Get the current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
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
            actor_loss = -self.critic.Q1(batch_s, self.actor(batch_s)).mean()  # Only use Q1
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
    model_name = "TD3_BUF"
    env = Env()
    # Set random seed
    seed = 10
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    state_dim = 12
    action_dim = 3
    max_action = 1.0
    max_episode_steps = 200  # Maximum number of steps per episode

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 2e6  # Maximum number of training steps
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration


    agent = TD3(state_dim, action_dim, max_action)
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
    # for t in range(int(max_train_steps)):
        episode_timesteps += 1

        if t < random_steps:  # Take the random actions in the beginning for the better exploration
            a = env.sample_action()
        else:
            # Add Gaussian noise to action for exploration
            a = agent.choose_action(s)
            a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)

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