# coding=utf-8
import torch
import matplotlib.pyplot as plt
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class OUNoise:
    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3, gain=0.01)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # self.l1 = nn.Linear(state_dim, 256)
        # self.l2 = nn.Linear(256 + action_dim, 256)
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3)

    def forward(self, state, action):
        # q = F.relu(self.l1(state))
        # q = F.relu(self.l2(torch.cat([q, action], dim=2)))
        q = F.relu(self.l1(torch.cat([state, action], dim=2)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.noise = OUNoise(action_dim)

    # CTDE，所以不存在train方法
    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten() + self.noise.noise()
        action = action.clip(-1, 1)
        return action

    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.actor_target.load_state_dict(params['actor_target'])
        self.critic_target.load_state_dict(params['critic_target'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
