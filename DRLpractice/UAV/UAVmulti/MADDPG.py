# coding=utf-8
from DRLpractice.UAV.UAVmulti.DDPGagent import DDPGAgent
import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = []
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 3e-4
        self.niter = 0
        for i in range(n_agents):
            self.agents.append(
                DDPGAgent(state_dim=12, action_dim=3, max_action=1.0,
                          discount=self.gamma, tau=self.tau, lr=self.lr))
        self.MseLoss = nn.MSELoss()

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
                                         for pi, nobs in zip(self.actor_targets, next_obs)]).to(device)
            all_trgt_Q = (rews[agent_i].view(-1, 1) + self.gamma *
                            curr_agent.critic_target(next_obs, all_trgt_acs) *
                            (1 - dones[agent_i].view(-1, 1)))  # tensor([2,64,1])
        # Compute the current Q and the critic loss
        current_Q = curr_agent.critic(obs, acs)  # tensor([2,64,1])
        critic_loss = self.MseLoss(current_Q, all_trgt_Q)
        # Optimize the critic
        curr_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # 更新Actor
        curr_pol_out = curr_agent.actor(obs[agent_i])   # tensor([64,3])
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i, pi, ob in zip(range(self.n_agents), self.actors, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in.cpu().data.numpy())
            else:
                all_pol_acs.append(pi(ob).cpu().data.numpy())
        # Compute the actor loss
        actor_loss = -curr_agent.critic(obs, torch.Tensor(all_pol_acs).to(device)).mean()
        # actor_loss += (curr_pol_out ** 2).mean() * 1e-3

        curr_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
        curr_agent.actor_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.critic_target, a.critic, self.tau)
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
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def load(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)

class MADDPGReplayBuffer(object):
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