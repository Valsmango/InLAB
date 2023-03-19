# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, s, h_0, c_0):
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

    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_width]), torch.zeros([1, batch_size, self.hidden_width])
        else:
            return torch.zeros([1, 1, self.hidden_width]), torch.zeros([1, 1, self.hidden_width])


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


class LSTMTD3Agent(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 64  # batch size

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.lr = 3e-4  # learning rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def choose_action(self, s, h, c):
        # s = torch.unsqueeze(s, 0)   # s = torch.FloatTensor(s.reshape(1, -1))
        s = torch.FloatTensor(s.reshape(1, -1)).unsqueeze(0).to(device)
        h = torch.FloatTensor(h).to(device)
        c = torch.FloatTensor(c).to(device)
        a, h, c = self.actor(s, h, c)
        return a.cpu().data.numpy().flatten(), h.cpu().data.numpy(), c.cpu().data.numpy()

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
