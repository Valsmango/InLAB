# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from DRLpractice.UAV.UAVmulti.envs.SimpleEnv import *
from torch.distributions import Normal
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
        self.n_agents = 2
        self.input_dim = self.n_agents * (state_dim + action_dim)
        # Q1
        self.l1 = nn.Linear(self.input_dim, hidden_width)
        # self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3)
        # Q2
        self.l4 = nn.Linear(self.input_dim, hidden_width)
        # self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)
        # orthogonal_init(self.l4)
        # orthogonal_init(self.l5)
        # orthogonal_init(self.l6)

    def forward(self, s, a):
        s_a = torch.cat([s, a], dim=2).view(-1, self.input_dim)
        # s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class MASACReplayBuffer(object):
    def __init__(self, n_agents, state_dim, action_dim, max_size):
        self.n_agents = n_agents
        self.max_size = max_size
        self.ptr = [0 for _ in range(self.n_agents)]
        self.size = [0 for _ in range(self.n_agents)]

        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.dones = []

        for i in range(n_agents):
            self.observations.append(np.zeros([max_size, state_dim]))
            self.actions.append(np.zeros([max_size, action_dim]))
            self.next_observations.append(np.zeros([max_size, state_dim]))
            self.rewards.append(np.zeros([max_size, 1]))
            self.dones.append(np.zeros([max_size, 1]))

    def add(self, observations, actions, next_observations, rewards, dones, flags):
        # 存入数据的格式：torch.tensor([n_agents, dim])
        for agent_i in range(self.n_agents):
            if flags[agent_i]:
                self.observations[agent_i][self.ptr] = np.stack(observations[agent_i, :])
                self.actions[agent_i][self.ptr] = np.stack(actions[agent_i, :])
                self.next_observations[agent_i][self.ptr] = np.stack(next_observations[agent_i, :])
                self.rewards[agent_i][self.ptr] = rewards[agent_i]
                self.dones[agent_i][self.ptr] = dones[agent_i]

                self.ptr[agent_i] = (self.ptr[agent_i] + 1) % self.max_size
                self.size[agent_i] = min(self.size[agent_i] + 1, self.max_size)

    def sample(self, batch_size):
        size = self.max_size
        for agent_i in range(self.n_agents):
            size = min(size, self.ptr[agent_i])
        index = np.random.choice(size, size=batch_size)  # Randomly sampling
        batch_obs = torch.tensor([self.observations[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_a = torch.tensor([self.actions[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_r = torch.tensor([self.rewards[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_next_obs = torch.tensor([self.next_observations[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_dw = torch.tensor([self.dones[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)

        return batch_obs, batch_a, batch_r, batch_next_obs, batch_dw

    def __len__(self) -> int:
        return self.size


class SACAgent(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 64  # batch size
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

    def choose_action(self, s, deterministic=False):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.cpu().data.numpy().flatten()


    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.critic_target.load_state_dict(params['critic_target'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class MASAC(object):
    def __init__(self, n_agents, state_dim, action_dim, max_action):
        self.n_agents = n_agents
        self.agents = []
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network

        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        for i in range(n_agents):
            self.agents.append(
                SACAgent(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action))
        self.MseLoss = nn.MSELoss()

    @property
    def actors(self):
        return [a.actor for a in self.agents]

    @property
    def actor_targets(self):
        return [a.actor_target for a in self.agents]

    def choose_actions(self, observations, deterministic=False):
        # return torch.Tensor([a.choose_action(obs)  for a, obs in zip(self.agents, observations)])
        actions = []
        for i in range(self.n_agents):
            a = self.agents[i].choose_action(observations[i], deterministic)
            actions.append(a)
        return torch.Tensor(actions)

    def update(self,  sample, agent_i, target_update=True):
        obs, acs, rews, next_obs, dones = sample

        curr_agent = self.agents[agent_i]

        with torch.no_grad():
            all_trgt_acs = []
            all_trgt_log_pi = []
            for i in range(self.n_agents):
                acs_next, log_pi_next = self.agents[i].actor(next_obs[i])  #acs_next为tensor([64,3])， log_pi_next为tensor([64,1])
                all_trgt_acs.append(acs_next.cpu().data.numpy())
                all_trgt_log_pi.append(log_pi_next.cpu().data.numpy())
            all_trgt_acs = torch.tensor(all_trgt_acs).to(device)  # tensor([2,64,3])
            all_trgt_log_pi = torch.tensor(all_trgt_log_pi).to(device)

            Q1_next, Q2_next = curr_agent.critic_target(next_obs, all_trgt_acs) # 输出为[2，64，1]
            all_trgt_Q = rews[agent_i] + self.GAMMA * (torch.min(Q1_next, Q2_next) - float(curr_agent.alpha) * all_trgt_log_pi[agent_i]) * (1 - dones[agent_i])

        # Compute current Q
        current_Q1, current_Q2 = curr_agent.critic(obs, acs)  # tensor([1,64,1])
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, all_trgt_Q) + F.mse_loss(current_Q2, all_trgt_Q)
        # Optimize the critic
        curr_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        curr_agent.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in curr_agent.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = curr_agent.actor(obs)
        Q1, Q2 = curr_agent.critic(obs, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (float(curr_agent.alpha) * log_pi - Q).mean()

        # Optimize the actor
        curr_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        curr_agent.actor_optimizer.step()

        # Unfreeze critic networks
        for params in curr_agent.critic.parameters():
            params.requires_grad = True

        # Update alpha
        if curr_agent.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(curr_agent.log_alpha.exp().to(device) * (log_pi + curr_agent.target_entropy).detach()).mean()
            curr_agent.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            curr_agent.alpha_optimizer.step()
            curr_agent.alpha = curr_agent.log_alpha.exp()

        # Softly update target
        if target_update:
            for param, target_param in zip(curr_agent.critic.parameters(), curr_agent.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def save(self, filename):
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def load(self, filename):
        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)


if __name__ == '__main__':
    start_time = time.time()
    env_name ="MAStandardEnv"
    env = Env(mode="train")
    policy_name = "MASAC"

    # Set random seed
    seed = 10
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    state_dim = 15
    action_dim = 3
    max_action = 1.0
    n_agents = 2
    max_episode_steps = 200  # Maximum number of steps per episode
    batch_size = 64

    max_timesteps = 1e6  # Maximum number of training steps
    start_timesteps = 25e3  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    train_episode_rewards = []
    train_episode_ma_rewards = []
    episode_reward = 0
    episode_reward_agents = [0 for _ in range(n_agents)]
    episode_timesteps = 0
    episode_num = 0
    update_target_freq = 100  # fixed target

    agent = MASAC(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    replay_buffer = MASACReplayBuffer(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_size=int(max_timesteps))

    if not os.path.exists("./eval_reward_train/MASAC/"):
        os.makedirs("./eval_reward_train/MASAC/")
    if not os.path.exists("./model_train/MASAC/"):
        os.makedirs("./model_train/MASAC/")

    # Evaluate untrained policy
    evaluate_rewards = []
    train_episode_rewards = []
    train_episode_ma_rewards = []
    evaluate_num = 0
    evaluate_freq = 5e3

    observations, dones = env.reset(), [False for i in range(n_agents)]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):

        episode_timesteps += 1
        if t < start_timesteps:
            actions = env.sample_action()
        else:
            # Add Gaussian noise to action for exploration
            actions = agent.choose_actions(observations=observations)

        # Perform action
        next_observations, rewards, dones, store_flags = env.step(actions)
        done_bools = [float(dones[i]) if episode_timesteps < env._max_episode_steps else 0 for i in range(n_agents)]

        replay_buffer.add(observations=observations, actions=actions, rewards=rewards,
                          next_observations=next_observations, dones=done_bools, flags=store_flags)
        observations = next_observations
        episode_reward += sum(rewards)

        if t >= start_timesteps:
            # if t % update_target_freq == 0:
            #     update_target = True
            # else:
            #     update_target = False
            update_target = True    # 不是fixed target
            for i in range(n_agents):
                sample = replay_buffer.sample(batch_size)
                agent.update(sample, i, update_target)

        if np.all(np.array(dones)):
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            train_episode_rewards.append(episode_reward)
            if train_episode_ma_rewards:
                train_episode_ma_rewards.append(0.99 * train_episode_ma_rewards[-1] + 0.01 * episode_reward)  # 移动平均，每100个episode的
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}  "
                      f"Sum: {episode_reward:.3f}  Avg: {0.99 * train_episode_ma_rewards[-1] + 0.01 * episode_reward:.3f}     "
                      f"Success:{env.success_count}")
            else:
                train_episode_ma_rewards.append(episode_reward)
            if (t+1) % evaluate_freq == 0:
                np.save('./eval_reward_train/MASAC/train_reward_MASAC_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_rewards))
                np.save('./eval_reward_train/MASAC/train_ma_reward_MASAC_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_ma_rewards))
            # Reset environment
            env.close()
            env = Env(mode="train")
            observations, dones = env.reset(), [False for i in range(n_agents)]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    env.close()
    agent.save(f"./model_train/MASAC/MASAC_{env_name}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}")