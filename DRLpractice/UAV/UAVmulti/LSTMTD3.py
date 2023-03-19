# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import copy
from DRLpractice.UAV.UAVmulti.LSTMTD3agent import LSTMTD3Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MALSTMTD3(object):
    def __init__(self, n_agents, state_dim, action_dim, max_action):
        self.n_agents = n_agents
        self.agents = []
        self.policy_freq = 2  # The frequency of policy updates
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network

        self.max_action = max_action
        self.policy_noise = 0.2 * self.max_action  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5 * self.max_action  # Clip the noise
        self.noise_std = 0.1 * self.max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor_pointer = 0
        for i in range(n_agents):
            self.agents.append(
                LSTMTD3Agent(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action))
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

    def update(self, sample, agent_i):
        obs, acs, rews, next_obs, dones = sample
        self.actor_pointer += 1
        # obs:tensor([2,64,12])  acs:tensor([2,64,3])  rews:tensor([2,64,1])  dones:tensor([2,64,1])
        curr_agent = self.agents[agent_i]
        # 更新Critic
        # Compute the target Q
        with torch.no_grad():
            all_trgt_acs = []
            for i in range(self.n_agents):
                h_target_0, c_target_0 = curr_agent.actor.init_hidden_state(batch_size=self.batch_size, training=True)
                h_target_0, c_target_0 = h_target_0.to(device), c_target_0.to(device)
                h_target_1, c_target_1 = curr_agent.critic.init_hidden_state(batch_size=self.batch_size, training=True)
                h_target_2, c_target_2 = copy.deepcopy(h_target_1), copy.deepcopy(c_target_1)
                h_target_1, c_target_1 = h_target_1.to(device), c_target_1.to(device)
                h_target_2, c_target_2 = h_target_2.to(device), c_target_2.to(device)
                acs_next, h_target_0, c_target_0 = self.agents[i].actor_target(next_obs[i], h_target_0, c_target_0)  # tensor([64,3])
                noise = (torch.randn_like(acs_next) * self.policy_noise).clamp(-self.noise_clip,
                                                                                   self.noise_clip) # tensor([64,3])
                acs_next = (acs_next + noise).clamp(-self.max_action, self.max_action)
                all_trgt_acs.append(acs_next.cpu().data.numpy())
            all_trgt_acs = torch.tensor(all_trgt_acs).to(device)  # tensor([2,64,3])

            Q1_next, Q2_next = curr_agent.critic_target(next_obs, all_trgt_acs)
            all_trgt_Q = rews[agent_i] + self.GAMMA * torch.min(Q1_next, Q2_next) * (1 - dones[agent_i])

        # Compute the current Q and the critic loss
        h_current, c_current = curr_agent.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current, c_current = h_current.to(device), c_current.to(device)
        h_current_, c_current_ = curr_agent.critic.init_hidden_state(batch_size=self.batch_size, training=True)
        h_current_, c_current_ = h_current_.to(device), c_current_.to(device)
        current_Q1, current_Q2, h_current, c_current, h_current_, c_current_ = \
            curr_agent.critic(obs, acs, h_current, c_current, h_current_, c_current_)  # tensor([2,64,1])
        critic_loss = self.MseLoss(current_Q1, all_trgt_Q) + self.MseLoss(current_Q2, all_trgt_Q)
        # Optimize the critic
        curr_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # 更新Actor
        if self.actor_pointer % (self.policy_freq * self.n_agents) == 0:
            h, c = curr_agent.actor.init_hidden_state(batch_size=batch_size, training=True)
            h, c = h.to(device), c.to(device)
            curr_pol_out, h, c = curr_agent.actor(obs[agent_i], h, c)  # tensor([64,3])
            curr_pol_vf_in = curr_pol_out
            all_pol_acs = []
            for i, pi, ob in zip(range(self.n_agents), self.actors, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in.cpu().data.numpy())
                else:
                    h_1, c_1 = curr_agent.critic.init_hidden_state(batch_size=self.batch_size, training=True)
                    h_1, c_1 = h_1.to(device), c_1.to(device)
                    temp_a, h_1, c_1 = pi(ob, h_1, c_1)
                    all_pol_acs.append(temp_a.cpu().data.numpy())
            # Compute the actor loss
            actor_loss = -curr_agent.critic.Q1(obs, torch.Tensor(all_pol_acs).to(device)).mean()
            # actor_loss += (curr_pol_out ** 2).mean() * 1e-3

            curr_agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
            curr_agent.actor_optimizer.step()

            self.soft_update(curr_agent.critic_target, curr_agent.critic, self.TAU)
            self.soft_update(curr_agent.actor_target, curr_agent.actor, self.TAU)

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


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, max_epi_num=20000, max_epi_len=200,
                 batch_size=1,
                 lookup_step=None):
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def _pre_sample(self):  # Sequential update
        sampled_buffer = []
        idx = np.random.randint(0, len(self.memory))
        sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def sample(self, batch_size):
        samples, seq_len = self._pre_sample()
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for i in range(batch_size):
            observations.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            rewards.append(samples[i]["rews"])
            next_observations.append(samples[i]["next_obs"])
            dones.append(samples[i]["done"])

        observations = torch.FloatTensor(observations).reshape(batch_size, seq_len, -1).to(device)
        actions = torch.FloatTensor(actions).reshape(batch_size, seq_len, -1).to(device)
        rewards = torch.FloatTensor(rewards).reshape(batch_size, seq_len, -1).to(device)
        next_observations = torch.FloatTensor(next_observations).reshape(batch_size, seq_len, -1).to(device)
        dones = torch.FloatTensor(dones).reshape(batch_size, seq_len, -1).to(device)
        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.memory)

class EpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(np.array(transition[0]))
        self.action.append(np.array(transition[1]))
        self.reward.append(np.array(transition[2]))
        self.next_obs.append(np.array(transition[3]))
        self.done.append(np.array(transition[4]))

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx + lookup_step]
            action = action[idx:idx + lookup_step]
            reward = reward[idx:idx + lookup_step]
            next_obs = next_obs[idx:idx + lookup_step]
            done = done[idx:idx + lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)