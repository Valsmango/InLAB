# coding=utf-8
"""
ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py

main：
    每一个episode：
        obs = env.reset()
        agent_actions = maddpg.step() # step也就是choose_action，一个maddpg中包括多个ddpg的agent，每个ddpg的agent去执行choose_action
        从里面提取出每个agent的action，然后进行格式转换（rearrange） --> actions
        next_obs, rewards, dones, infos = env.step(actions)
        replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
        obs = next_obs

"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from DRLpractice.UAV.UAVmulti.envs.Env import Env

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
        self.n_agents = 2
        self.input_dim = self.n_agents*(state_dim + action_dim)
        # self.l1 = nn.Linear(state_dim, 256)
        # self.l2 = nn.Linear(256 + action_dim, 256)
        self.l1 = nn.Linear(self.input_dim, 256)
        # self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # orthogonal_init(self.l1)
        # orthogonal_init(self.l2)
        # orthogonal_init(self.l3)

    def forward(self, state, action):
        # 输入：【agent，batch，dim】 -- > cat之后【agent，batch，s+a的dim】
        # q = F.relu(self.l1(state))
        # q = F.relu(self.l2(torch.cat([q, action], dim=2)))
        x = torch.cat([state, action], dim=2).view(-1, self.input_dim)
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)

        # self.critic = Critic(state_dim, action_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

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

    def scale_noise(self, scale):
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def get_params(self):
        return {'actor': self.actor.state_dict(),
                # 'critic': self.critic.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                # 'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                # 'critic_optimizer': self.critic_optimizer.state_dict()
                }

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        # self.critic.load_state_dict(params['critic'])
        self.actor_target.load_state_dict(params['actor_target'])
        # self.critic_target.load_state_dict(params['critic_target'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])


def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class MADDPG(object):
    def __init__(self, n_agents, state_dim, action_dim, max_action):
        self.n_agents = n_agents
        self.agents = []
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 3e-4
        self.niter = 0
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = max_action
        for i in range(n_agents):
            self.agents.append(
                DDPGAgent(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action,
                          discount=self.gamma, tau=self.tau, lr=self.lr))
        self.MseLoss = nn.MSELoss()
        self.common_critic = Critic(self.state_dim, self.action_dim).to(device)
        self.common_critic_target = copy.deepcopy(self.common_critic)
        self.common_critic_optimizer = torch.optim.Adam(self.common_critic.parameters(), self.lr)

    @property
    def actors(self):
        return [a.actor for a in self.agents]

    @property
    def actor_targets(self):
        return [a.actor_target for a in self.agents]

    def choose_actions(self, observations):
        return torch.Tensor([a.choose_action(obs) for a, obs in zip(self.agents, observations)])

    def update(self, sample, agent_i):
        obs, acs, rews, next_obs, dones = sample
        # obs:tensor([2,64,12])  acs:tensor([2,64,3])  rews:tensor([2,64,1])  dones:tensor([2,64,1])
        curr_agent = self.agents[agent_i]

        # 更新Critic
        # Compute the target Q
        with torch.no_grad():
            all_trgt_acs = torch.Tensor([pi(nobs).cpu().data.numpy()
                                         for pi, nobs in zip(self.actor_targets, next_obs)]).to(device)  # [2,64,3]
            all_trgt_Q = (rews[agent_i].view(-1, 1) + self.gamma *
                          self.common_critic(next_obs, all_trgt_acs) *
                          (1 - dones[agent_i].view(-1, 1)))  # reward和dones-->view:[64,1], 而next_obs和acs为tensor([2,64,1])
        # Compute the current Q and the critic loss
        current_Q = self.common_critic(obs, acs)  # obs和acs为tensor([2,64,1])
        critic_loss = self.MseLoss(current_Q, all_trgt_Q)
        # Optimize the critic
        self.common_critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        self.common_critic_optimizer.step()

        # 更新Actor
        curr_pol_out = curr_agent.actor(obs[agent_i])  # tensor([64,3])
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i, pi, ob in zip(range(self.n_agents), self.actors, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in.cpu().data.numpy())
            else:
                all_pol_acs.append(pi(ob).cpu().data.numpy())
        # Compute the actor loss
        actor_loss = -self.common_critic(obs, torch.Tensor(all_pol_acs).to(device)).mean()
        actor_loss += (curr_pol_out ** 2).mean() * 1e-3

        curr_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
        curr_agent.actor_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.common_critic_target, self.common_critic, self.tau)
        for a in self.agents:
            # soft_update(a.common_critic_target, a.critic, self.tau)
            soft_update(a.actor_target, a.actor, self.tau)
        self.niter += 1

    def scale_all_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_all_noise(self):
        for a in self.agents:
            a.reset_noise()

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        common_critic_dict = {'critic': self.common_critic.state_dict(),
                              'critic_target': self.common_critic_target.state_dict(),
                              'critic_optimizer': self.common_critic_optimizer.state_dict()}
        save_dict = {'agent_params': [a.get_params() for a in self.agents],
                     'common_critic': common_critic_dict}
        torch.save(save_dict, filename)

    def load(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)
        critic_params = save_dict['common_critic']
        self.common_critic.load_state_dict(critic_params['critic'])
        self.common_critic_target.load_state_dict(critic_params['critic_target'])
        self.common_critic_optimizer.load_state_dict(critic_params['critic_optimizer'])


# class MADDPGReplayBuffer(object):
#     def __init__(self, n_agents, state_dim, action_dim, max_size):
#         self.n_agents = n_agents
#         self.max_size = max_size
#         self.ptr = [0 for _ in range(self.n_agents)]
#         self.size = [0 for _ in range(self.n_agents)]
#
#         self.observations = []
#         self.actions = []
#         self.next_observations = []
#         self.rewards = []
#         self.dones = []
#
#         for i in range(n_agents):
#             self.observations.append(np.zeros([max_size, state_dim]))
#             self.actions.append(np.zeros([max_size, action_dim]))
#             self.next_observations.append(np.zeros([max_size, state_dim]))
#             self.rewards.append(np.zeros([max_size, 1]))
#             self.dones.append(np.zeros([max_size, 1]))
#
#         self.fail_ptr = [0 for _ in range(self.n_agents)]
#         self.fail_size = [0 for _ in range(self.n_agents)]
#         self.fail_max_size = int(max_size * 0.1)
#         self.fail_observations = []
#         self.fail_actions = []
#         self.fail_next_observations = []
#         self.fail_rewards = []
#         self.fail_dones = []
#         for i in range(n_agents):
#             self.fail_observations.append(np.zeros([self.fail_max_size, state_dim]))
#             self.fail_actions.append(np.zeros([self.fail_max_size, action_dim]))
#             self.fail_next_observations.append(np.zeros([self.fail_max_size, state_dim]))
#             self.fail_rewards.append(np.zeros([self.fail_max_size, 1]))
#             self.fail_dones.append(np.zeros([self.fail_max_size, 1]))
#
#     def add(self, observations, actions, next_observations, rewards, dones, flags):
#         # 存入数据的格式：torch.tensor([n_agents, dim])
#         for agent_i in range(self.n_agents):
#             if flags[agent_i]:
#                 self.observations[agent_i][self.ptr] = np.stack(observations[agent_i, :])
#                 self.actions[agent_i][self.ptr] = np.stack(actions[agent_i, :])
#                 self.next_observations[agent_i][self.ptr] = np.stack(next_observations[agent_i, :])
#                 self.rewards[agent_i][self.ptr] = rewards[agent_i]
#                 self.dones[agent_i][self.ptr] = dones[agent_i]
#
#                 self.ptr[agent_i] = (self.ptr[agent_i] + 1) % self.max_size
#                 self.size[agent_i] = min(self.size[agent_i] + 1, self.max_size)
#
#                 if dones[agent_i]:  # 不完全是fail，也有可能是成功的
#                     self.fail_observations[agent_i][self.fail_ptr] = np.stack(observations[agent_i, :])
#                     self.fail_actions[agent_i][self.fail_ptr] = np.stack(actions[agent_i, :])
#                     self.fail_next_observations[agent_i][self.fail_ptr] = np.stack(next_observations[agent_i, :])
#                     self.fail_rewards[agent_i][self.fail_ptr] = rewards[agent_i]
#                     self.fail_dones[agent_i][self.fail_ptr] = dones[agent_i]
#                     self.fail_ptr[agent_i] = (self.fail_ptr[agent_i] + 1) % self.fail_max_size
#                     self.fail_size[agent_i] = min(self.fail_size[agent_i] + 1, self.fail_max_size)
#
#     def sample(self, batch_size):
#         sample_size = int(batch_size * 0.9)
#         size = self.max_size
#         for agent_i in range(self.n_agents):
#             size = min(size, self.ptr[agent_i])
#         index = np.random.choice(size, size=sample_size)  # Randomly sampling
#         batch_obs = torch.tensor([self.observations[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_a = torch.tensor([self.actions[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_r = torch.tensor([self.rewards[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_next_obs = torch.tensor([self.next_observations[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_dw = torch.tensor([self.dones[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float)
#
#         fail_size = self.fail_max_size
#         for agent_i in range(self.n_agents):
#             fail_size = min(size, self.fail_ptr[agent_i])
#         fail_index = np.random.choice(fail_size, size=batch_size-sample_size)  # Randomly sampling
#         batch_fail_obs = torch.tensor([self.fail_observations[agent_i][fail_index]
#                                        for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_fail_a = torch.tensor([self.fail_actions[agent_i][fail_index] for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_fail_r = torch.tensor([self.fail_rewards[agent_i][fail_index] for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_fail_next_obs = torch.tensor([self.fail_next_observations[agent_i][fail_index]
#                                             for agent_i in range(self.n_agents)], dtype=torch.float)
#         batch_fail_dw = torch.tensor([self.fail_dones[agent_i][fail_index] for agent_i in range(self.n_agents)], dtype=torch.float)
#
#         batch_obs = torch.cat((batch_obs, batch_fail_obs), dim=1).to(device)
#         batch_a = torch.cat((batch_a, batch_fail_a), dim=1).to(device)
#         batch_r = torch.cat((batch_r, batch_fail_r), dim=1).to(device)
#         batch_next_obs = torch.cat((batch_next_obs, batch_fail_next_obs), dim=1).to(device)
#         batch_dw = torch.cat((batch_dw, batch_fail_dw), dim=1).to(device)
#
#         # 返回之前先转换成tensor
#         # batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
#         # batch_a = torch.tensor(batch_a, dtype=torch.float).to(device)
#         # batch_r = torch.tensor(batch_r, dtype=torch.float).to(device)
#         # batch_next_obs = torch.tensor(batch_next_obs, dtype=torch.float).to(device)
#         # batch_dw = torch.tensor(batch_dw, dtype=torch.float).to(device)
#
#         return batch_obs, batch_a, batch_r, batch_next_obs, batch_dw
#
#     def __len__(self) -> int:
#         return self.size


# 普通的buffer
class MADDPGReplayBuffer(object):
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
        if np.all(np.array(flags)):
            for agent_i in range(self.n_agents):
                self.observations[agent_i][self.ptr] = np.stack(observations[agent_i, :])
                self.actions[agent_i][self.ptr] = np.stack(actions[agent_i, :])
                self.next_observations[agent_i][self.ptr] = np.stack(next_observations[agent_i, :])
                self.rewards[agent_i][self.ptr] = rewards[agent_i]
                self.dones[agent_i][self.ptr] = dones[agent_i]

                self.ptr[agent_i] = (self.ptr[agent_i] + 1) % self.max_size
                self.size[agent_i] = min(self.size[agent_i] + 1, self.max_size)
        # for agent_i in range(self.n_agents):
        #     if flags[agent_i]:
        #         self.observations[agent_i][self.ptr] = np.stack(observations[agent_i, :])
        #         self.actions[agent_i][self.ptr] = np.stack(actions[agent_i, :])
        #         self.next_observations[agent_i][self.ptr] = np.stack(next_observations[agent_i, :])
        #         self.rewards[agent_i][self.ptr] = rewards[agent_i]
        #         self.dones[agent_i][self.ptr] = dones[agent_i]
        #
        #         self.ptr[agent_i] = (self.ptr[agent_i] + 1) % self.max_size
        #         self.size[agent_i] = min(self.size[agent_i] + 1, self.max_size)

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


def eval_policy(maddpg):
    times = 40  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    n_agents = 2
    # seed = 20
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    for _ in range(times):
        env = Env(mode="test")
        observations = env.reset()
        dones = [False for i in range(n_agents)]
        episode_reward = 0
        while not np.any(np.array(dones)):
            actions = maddpg.choose_actions(observations=observations)
            next_observations, rewards, dones, flags = env.step(actions)
            episode_reward += sum(rewards)
            observations = next_observations
        env.close()
        evaluate_reward += episode_reward

    return evaluate_reward / times


if __name__ == "__main__":

    starttime = time.time()
    eval_records = []

    seed = 10
    env_name = "MAStandardEnv"
    env = Env(mode="train")
    policy_name = "MADDPG"
    state_dim = 15
    action_dim = 3
    max_action = 1.0
    n_agents = 2
    max_timesteps = 1e6
    start_timesteps = 25e3
    eval_freq = 5e3
    expl_noise = 0.1  # Std of Gaussian exploration noise
    batch_size = 64

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    file_name = f"{policy_name}_env_{env_name}_seed_{seed}"
    print("---------------------------------------")
    print(f"Policy: {policy_name}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    if not os.path.exists("./eval_reward_train/MADDPG/"):
        os.makedirs("./eval_reward_train/MADDPG/")

    if not os.path.exists("./model_train/MADDPG/"):
        os.makedirs("./model_train/MADDPG/")

    maddpg = MADDPG(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    replay_buffer = MADDPGReplayBuffer(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_size=int(max_timesteps))
    # explr_pct_remaining = max(0, n_exploration_eps - ep_i) / config.n_exploration_eps
    # maddpg.scale_noise(
    #     config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
    # maddpg.reset_noise()

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

    update_iter_count = 0
    update_freq = 2   # Frequency of delayed policy updates

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # env.render()
        # 初始不再是xxx步用于随机探索（即Select action randomly or according to policy）
        # 全都是根据policy
        if t < start_timesteps:
            actions = env.sample_action()
        else:
            actions = maddpg.choose_actions(observations=observations)

        # Perform action
        next_observations, rewards, dones, store_flags = env.step(actions)
        done_bools = [float(dones[i]) if episode_timesteps < env._max_episode_steps else 0 for i in range(n_agents)]

        # Store data in replay buffer
        replay_buffer.add(observations=observations, actions=actions, rewards=rewards,
                          next_observations=next_observations, dones=done_bools, flags=store_flags)
        observations = next_observations
        episode_reward += sum(rewards)

        # Train agent after collecting sufficient data
        # 只训练了一次，DDPG那个代码里面是隔50步训练50次……但标准的TD3其实也是走一步训练一次
        if t >= start_timesteps:
            update_iter_count += 1
            for agent_i in range(n_agents):
                sample = replay_buffer.sample(batch_size)
                maddpg.update(sample, agent_i)
            if update_iter_count % update_freq == 0:
                maddpg.update_all_targets()

        # # if t >= start_timesteps and (t + 1) % evaluate_freq == 0:
        # if (t + 1) % evaluate_freq == 0:
        #     evaluate_num += 1
        #     evaluate_reward = eval_policy(maddpg)
        #     evaluate_rewards.append(evaluate_reward)
        #     # print("---------------------------------------")
        #     print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
        #     # print("---------------------------------------")
        #     # writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
        #     # Save the rewards
        #     if evaluate_num % 10 == 0:
        #         np.save('./eval_reward_train/MADDPG/MADDPG_env_{}_seed_{}.npy'.format(env_name, seed),
        #                 np.array(evaluate_rewards))

        if np.all(np.array(dones)):
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            train_episode_rewards.append(episode_reward)
            if train_episode_ma_rewards:
                train_episode_ma_rewards.append(
                    0.98 * train_episode_ma_rewards[-1] + 0.02 * episode_reward)  # 移动平均，每50个episode的
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}  "
                      f"Sum: {episode_reward:.3f}  Avg: {0.99 * train_episode_ma_rewards[-1] + 0.01 * episode_reward:.3f}     "
                      f"Success:{env.success_count}")
            else:
                train_episode_ma_rewards.append(episode_reward)
            if (t + 1) % evaluate_freq == 0:
                np.save('./eval_reward_train/MADDPG/train_reward_MADDPG_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_rewards))
                np.save('./eval_reward_train/MADDPG/train_ma_reward_MADDPG_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_ma_rewards))
            # Reset environment
            env.close()
            env = Env(mode="train")
            observations, done = env.reset(), [False for i in range(n_agents)]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    env.close()
    maddpg.save(f"./model_train/MADDPG/MADDPG_{env_name}")

    endtime = time.time()
    dtime = endtime - starttime
    end_time = time.time()
    print("程序运行时间：%.8s s" % dtime)
