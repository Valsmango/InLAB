# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
from DRLpractice.UAV.UAVmulti.TD3agent import TD3Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MATD3(object):
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = []
        self.policy_freq = 2  # The frequency of policy updates
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network

        self.max_action = 1.0
        self.policy_noise = 0.2 * self.max_action  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5 * self.max_action  # Clip the noise
        self.noise_std = 0.1 * self.max_action
        self.action_dim = 3
        self.actor_pointer = 0
        for i in range(n_agents):
            self.agents.append(
                TD3Agent(state_dim=12, action_dim=self.action_dim, max_action=self.max_action))
        self.MseLoss = nn.MSELoss()

    @property
    def actors(self):
        return [a.actor for a in self.agents]

    @property
    def actor_targets(self):
        return [a.actor_target for a in self.agents]

    def choose_actions(self, observations):
        # return torch.Tensor([a.choose_action(obs)  for a, obs in zip(self.agents, observations)])
        actions = []
        for i in range(self.n_agents):
            a = self.agents[i].choose_action(observations[i])
            a = (a + np.random.normal(0, self.noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
            actions.append(a)
        return torch.Tensor(actions)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, sample):
        # 和DDPG代码不同的一点在于这里的训练所有agent用的同一组sample
        obs, acs, rews, next_obs, dones = sample
        self.actor_pointer += 1
        # obs:tensor([2,64,12])  acs:tensor([2,64,3])  rews:tensor([2,64,1])  dones:tensor([2,64,1])
        for agent_i in range(self.n_agents):
            curr_agent = self.agents[agent_i]

            # 更新Critic
            # Compute the target Q
            with torch.no_grad():
                all_trgt_acs = []
                for i in range(self.n_agents):
                    acs_next = self.agents[i].actor_target(next_obs[i])  # tensor([64,3])
                    noise = (torch.randn_like(acs_next) * self.policy_noise).clamp(-self.noise_clip,
                                                                                       self.noise_clip) # tensor([64,3])
                    acs_next = (acs_next + noise).clamp(-self.max_action, self.max_action)
                    all_trgt_acs.append(acs_next.cpu().data.numpy())
                all_trgt_acs = torch.tensor(all_trgt_acs).to(device)  # tensor([2,64,3])

                Q1_next, Q2_next = curr_agent.critic_target(next_obs, all_trgt_acs)
                all_trgt_Q = rews[agent_i] + self.GAMMA * torch.min(Q1_next, Q2_next) * (1 - dones[agent_i])

            # Compute the current Q and the critic loss
            current_Q1, current_Q2 = curr_agent.critic(obs, acs)  # tensor([2,64,1])
            critic_loss = self.MseLoss(current_Q1, all_trgt_Q) + self.MseLoss(current_Q2, all_trgt_Q)
            # Optimize the critic
            curr_agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
            curr_agent.critic_optimizer.step()

            # 更新Actor
            if self.actor_pointer % self.policy_freq == 0:
                curr_pol_out = curr_agent.actor(obs[agent_i])  # tensor([64,3])
                curr_pol_vf_in = curr_pol_out
                all_pol_acs = []
                for i, pi, ob in zip(range(self.n_agents), self.actors, obs):
                    if i == agent_i:
                        all_pol_acs.append(curr_pol_vf_in.cpu().data.numpy())
                    else:
                        all_pol_acs.append(pi(ob).cpu().data.numpy())
                # Compute the actor loss
                actor_loss = -curr_agent.critic.Q1(obs, torch.Tensor(all_pol_acs).to(device)).mean()
                # actor_loss += (curr_pol_out ** 2).mean() * 1e-3

                curr_agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                # nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
                curr_agent.actor_optimizer.step()

                self.soft_update(curr_agent.critic_target, curr_agent.critic, self.TAU)
                self.soft_update(curr_agent.actor_target, curr_agent.actor, self.TAU)


    def scale_all_noise(self, scale):
        for a in self.agents:
            a.scale_noise(scale)

    def reset_all_noise(self):
        for a in self.agents:
            a.reset_noise()

    def save(self, filename):
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def load(self, filename):
        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)


class MATD3ReplayBuffer(object):
    def __init__(self, n_agents, state_dim, action_dim, max_size):
        self.n_agents = n_agents
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

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

    def add(self, observations, actions, next_observations, rewards, dones):
        # 存入数据的格式：torch.tensor([n_agents, dim])
        for agent_i in range(self.n_agents):
            self.observations[agent_i][self.ptr] = np.stack(observations[agent_i, :])
            self.actions[agent_i][self.ptr] = np.stack(actions[agent_i, :])
            self.next_observations[agent_i][self.ptr] = np.stack(next_observations[agent_i, :])
            self.rewards[agent_i][self.ptr] = rewards[agent_i]
            self.dones[agent_i][self.ptr] = dones[agent_i]

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_obs = [np.array(self.observations[agent_i][index]) for agent_i in range(self.n_agents)]
        batch_a = [self.actions[agent_i][index] for agent_i in range(self.n_agents)]
        batch_r = [self.rewards[agent_i][index] for agent_i in range(self.n_agents)]
        batch_next_obs = [self.next_observations[agent_i][index] for agent_i in range(self.n_agents)]
        batch_dw = [self.dones[agent_i][index] for agent_i in range(self.n_agents)]

        # 返回之前先转换成tensor
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
        batch_a = torch.tensor(batch_a, dtype=torch.float).to(device)
        batch_r = torch.tensor(batch_r, dtype=torch.float).to(device)
        batch_next_obs = torch.tensor(batch_next_obs, dtype=torch.float).to(device)
        batch_dw = torch.tensor(batch_dw, dtype=torch.float).to(device)

        return batch_obs, batch_a, batch_r, batch_next_obs, batch_dw

    def __len__(self) -> int:
        return self.size