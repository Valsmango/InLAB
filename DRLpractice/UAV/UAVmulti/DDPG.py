# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from DRLpractice.UAV.UAVmulti.envs.Env import *
import os
import time

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
        # 输入：【agent，batch，dim】 -- > cat之后【agent，batch，s+a的dim】
        # q = F.relu(self.l1(state))
        # q = F.relu(self.l2(torch.cat([q, action], dim=2)))
        x = torch.cat([state, action], dim=1)
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        return self.l3(q)


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

    def store(self, s, a, r, s_, dw, store_flag):
        if store_flag:
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


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, lr=3e-4):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.discount = discount
        self.tau = tau
        self.batch_size = 64

        self.noise = OUNoise(action_dim)
        self.MseLoss = nn.MSELoss()

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten() + self.noise.noise()
        action = action.clip(-1, 1)
        return action

    def learn(self, replay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = replay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.discount * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()
                }

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.actor_target.load_state_dict(params['actor_target'])
        self.critic_target.load_state_dict(params['critic_target'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


if __name__ == '__main__':
    start_time = time.time()
    env_name ="MAStandardEnv"
    policy_name = "DDPG"
    # Set random seed
    seed = 10
    # env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_dim = 15
    action_dim = 3
    max_action = 1.0
    n_agents = 3
    max_episode_steps = 200  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 1e6  # Maximum number of training steps
    # max_train_episodes = 2e4
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    max_train_episodes = 2e4
    # evaluate_num = 0  # Record the number of evaluations
    # evaluate_rewards = []  # Record the rewards during the evaluating
    train_episode_rewards = []
    train_episode_ma_rewards = []
    episode_reward = 0
    episode_reward_agents = [0 for _ in range(n_agents)]
    episode_timesteps = 0
    episode_num = 0
    train_episode_success_rate = []
    # train_episode_collision_rate = []
    train_episode_ma_success_rate = []
    # train_episode_ma_collision_rate = []

    agent = [DDPG(state_dim, action_dim, max_action) for _ in range(n_agents)]
    replay_buffer = [ReplayBuffer(buffer_capacity=int(max_train_steps), state_dim=state_dim, action_dim=action_dim) for _ in range(n_agents)]
    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/TD3/TD3_env_{}_seed_{}'.format(env_name, seed))

    if not os.path.exists("./eval_reward_train/DDPG/"):
        os.makedirs("./eval_reward_train/DDPG/")
    if not os.path.exists("model_train/DDPG/"):
        os.makedirs("./model_train/DDPG/")

    env = Env(mode="train")
    s, done = env.reset(), [False for i in range(n_agents)]

    t = 0
    while episode_num < max_train_episodes:
    # for t in range(int(max_train_steps)):

        episode_timesteps += 1
        if t < random_steps:  # Take random actions in the beginning for the better exploration
            a = env.sample_action()
        else:
            a = [agent[i].choose_action(s[i]) for i in range(n_agents)]

        s_, r, done, store_flags = env.step(a)
        for i in range(n_agents):
            dw = float(done[i]) if episode_timesteps < env._max_episode_steps else 0
            replay_buffer[i].store(s[i], a[i], r[i], s_[i], dw, store_flags[i])  # Store the transition
        s = s_
        episode_reward += sum(r)
        for i in range(n_agents):
            episode_reward_agents[i] += r[i]

        if t >= random_steps:
            for i in range(n_agents):
                agent[i].learn(replay_buffer[i])

        if np.all(np.array(done)):
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            train_episode_rewards.append(episode_reward)
            train_episode_success_rate.append(env.success_count/n_agents)
            if train_episode_ma_rewards:
                train_episode_ma_rewards.append(0.99 * train_episode_ma_rewards[-1] + 0.01 * episode_reward)  # 移动平均，每100个episode的
                train_episode_ma_success_rate.append(
                    0.99 * train_episode_ma_success_rate[-1] + 0.01 * env.success_count/n_agents)
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}  "
                      f"Reward: {episode_reward_agents[0]:.3f} {episode_reward_agents[1]:.3f}  Sum: {episode_reward:.3f}  "
                      f"Avg: {0.99 * train_episode_ma_rewards[-1] + 0.01 * episode_reward:.3f}     "
                      f"Success:{env.success_count}")
            else:
                train_episode_ma_rewards.append(episode_reward)
                train_episode_ma_success_rate.append(env.success_count/n_agents)
            if (episode_num + 1) % 100 == 0:
                np.save('./eval_reward_train/DDPG/train_reward_DDPG_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_rewards))
                np.save('./eval_reward_train/DDPG/train_ma_reward_DDPG_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_ma_rewards))
                np.save('./eval_reward_train/DDPG/train_success_DDPG_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_success_rate))
                np.save('./eval_reward_train/DDPG/train_ma_success_DDPG_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_ma_success_rate))

            # Reset environment
            env.close()
            env = Env(mode="train")
            s, done = env.reset(), [False for i in range(n_agents)]
            episode_reward = 0
            episode_reward_agents = [0 for _ in range(n_agents)]
            episode_timesteps = 0
            episode_num += 1

        t += 1


    ######################### DDPG save没写  ############################


    env.close()
    for i in range(n_agents):
        agent[i].save(f"./model_train/DDPG/agent_{i}_DDPG_{env_name}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}")