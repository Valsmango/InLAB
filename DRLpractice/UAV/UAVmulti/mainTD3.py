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


# def evaluate_policy(agent):
#     times = 40  # Perform three evaluations and calculate the average
#     evaluate_reward = 0
#     for _ in range(times):
#         env = Env()
#         s = env.reset()
#         done = False
#         episode_reward = 0
#         while not done:
#             a = agent.choose_action(s)  # We do not add noise when evaluating
#             s_, r, done = env.step(a)
#             episode_reward += r
#             s = s_
#         env.close()
#         evaluate_reward += episode_reward
#
#     return evaluate_reward / times

if __name__ == '__main__':
    start_time = time.time()
    env_name ="MAStandardEnv"
    policy_name = "TD3"
    # Set random seed
    seed = 10
    # env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_dim = 18
    action_dim = 3
    max_action = 1.0
    n_agents = 2
    max_episode_steps = 200  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 2e6  # Maximum number of training steps
    # max_train_episodes = 2e4
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    train_episode_rewards = []
    train_episode_ma_rewards = []
    episode_reward = 0
    episode_reward_agents = [0 for _ in range(n_agents)]
    episode_timesteps = 0
    episode_num = 0

    agent = [TD3(state_dim, action_dim, max_action) for _ in range(n_agents)]
    replay_buffer = [ReplayBuffer(buffer_capacity=int(max_train_steps), state_dim=state_dim, action_dim=action_dim) for _ in range(n_agents)]
    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/TD3/TD3_env_{}_seed_{}'.format(env_name, seed))

    if not os.path.exists("./eval_reward_train/TD3/"):
        os.makedirs("./eval_reward_train/TD3/")
    if not os.path.exists("model_train/TD3_07_random_start/"):
        os.makedirs("./model_train/TD3/")

    env = Env(mode="train")
    s, done = env.reset(), [False for i in range(n_agents)]
    for t in range(int(max_train_steps)):

        episode_timesteps += 1
        if t < random_steps:  # Take random actions in the beginning for the better exploration
            a = env.sample_action()
        else:
            # Add Gaussian noise to action for exploration
            a = [(agent[i].choose_action(s[i]) +
                  np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                 for i in range(n_agents)]

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
            if train_episode_ma_rewards:
                train_episode_ma_rewards.append(0.99 * train_episode_ma_rewards[-1] + 0.01 * episode_reward)  # 移动平均，每100个episode的
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}  "
                      f"Reward: {episode_reward_agents[0]:.3f} {episode_reward_agents[1]:.3f}  Sum: {episode_reward:.3f}  "
                      f"Avg: {0.99 * train_episode_ma_rewards[-1] + 0.01 * episode_reward:.3f}     "
                      f"Success:{env.success_count}")
            else:
                train_episode_ma_rewards.append(episode_reward)
            if (t+1) % evaluate_freq == 0:
                np.save('./eval_reward_train/TD3/train_reward_TD3_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_rewards))
                np.save('./eval_reward_train/TD3/train_ma_reward_TD3_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_ma_rewards))
            # Reset environment
            env.close()
            env = Env(mode="train")
            s, done = env.reset(), [False for i in range(n_agents)]
            episode_reward = 0
            episode_reward_agents = [0 for _ in range(n_agents)]
            episode_timesteps = 0
            episode_num += 1

        env.close()
    for i in range(n_agents):
        agent[i].save(f"./model_train/TD3/agent_{i}_TD3_{env_name}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}")