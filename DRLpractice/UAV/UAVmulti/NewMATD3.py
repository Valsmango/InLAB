# coding=utf-8
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from DRLpractice.UAV.UAVmulti.envs.Env import Env


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.n_agents = 3
        self.input_dim = self.n_agents * (state_dim + action_dim)
        # self.input_dim = self.n_agents * (9 + 3)
        # Q1
        self.l1 = nn.Linear(self.input_dim, hidden_width)
        # self.l2 = nn.Linear(state_dim + action_dim, hidden_width)
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
        # s_a = torch.cat([s, a], dim=2)
        # s = s[:, :, 0:9]
        s_a = torch.cat([s, a], dim=2).view(-1, self.input_dim)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, s, a):
        # s_a = torch.cat([s, a], 2)
        # s = s[:, :, 0:9]
        s_a = torch.cat([s, a], dim=2).view(-1, self.input_dim)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3Agent(object):
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

    def choose_action(self, s):
        # s = torch.unsqueeze(s, 0)   # s = torch.FloatTensor(s.reshape(1, -1))
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        a = self.actor(s).cpu().data.numpy().flatten()
        return a

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


class MATD3(object):
    def __init__(self, n_agents, state_dim, action_dim, max_action):
        self.n_agents = n_agents
        self.agents = []
        self.policy_freq = 2  # The frequency of policy updates
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network

        self.max_action = max_action
        self.policy_noise = 0.2 * self.max_action  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5 * self.max_action  # Clip the noise
        self.noise_std = 0.3 * self.max_action  # 原本是0.1
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor_pointer = 0
        for i in range(n_agents):
            self.agents.append(
                TD3Agent(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action))
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

    def update(self, sample, agent_i, target_update):
        obs, acs, rews, next_obs, dones = sample
        self.actor_pointer += 1
        # obs:tensor([2,64,12])  acs:tensor([2,64,3])  rews:tensor([2,64,1])  dones:tensor([2,64,1])
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

            Q1_next, Q2_next = curr_agent.critic_target(next_obs, all_trgt_acs)  # Q1和Q2都是tensor([2,64,1])
            # torch.min(Q1,Q2)也是tensor([2,64,1])
            all_trgt_Q = rews[agent_i] + self.GAMMA * torch.min(Q1_next, Q2_next) * (1 - dones[agent_i])    # tensor([2,64,1])

        # Compute the current Q and the critic loss
        current_Q1, current_Q2 = curr_agent.critic(obs, acs)  # Q1和Q2都是tensor([2,64,1])
        critic_loss = self.MseLoss(current_Q1, all_trgt_Q) + self.MseLoss(current_Q2, all_trgt_Q)
        # Optimize the critic
        curr_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()


        if (self.actor_pointer - 1) % (self.policy_freq*self.n_agents) == agent_i:
            # 更新Actor
            ############################ 更新方式1：全体最优 ####################################
            # curr_pol_out = curr_agent.actor(obs[agent_i])  # tensor([64,3])
            # curr_pol_vf_in = curr_pol_out
            # all_pol_acs = []
            # for i, pi, ob in zip(range(self.n_agents), self.actors, obs):
            #     if i == agent_i:
            #         all_pol_acs.append(curr_pol_vf_in.cpu().data.numpy())
            #     else:
            #         all_pol_acs.append(pi(ob).cpu().data.numpy())
            # # Compute the actor loss
            # # 假设别人仍以当前策略行动，而自己要取得最优动作
            # actor_loss = -curr_agent.critic.Q1(obs, torch.Tensor(all_pol_acs).to(device)).mean()
            # # actor_loss += (curr_pol_out ** 2).mean() * 1e-3

            # 尝试增加个体最优，mask掉其他人的策略行动，只关注自己的动作、自己的state
            # extra_self_obs = torch.zeros_like(obs)
            # extra_self_acs = np.zeros_like(all_pol_acs)
            # extra_self_obs[agent_i] = obs[agent_i]
            # extra_self_acs[agent_i] = all_pol_acs[agent_i]
            # actor_loss += -curr_agent.critic.Q1(extra_self_obs, torch.Tensor(extra_self_acs).to(device)).mean()

            # ############################ 更新方式2：个体最优 ####################################
            # # 使agent自身的Q最大，即以个体最优为目标：
            actor_loss = -curr_agent.critic.Q1(obs[agent_i].unsqueeze(0), curr_agent.actor(obs[agent_i]).unsqueeze(0)).mean()

            curr_agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
            curr_agent.actor_optimizer.step()

        if target_update:
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
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        for agent_i in range(self.n_agents):
            if flags[agent_i]:
                self.observations[agent_i][self.ptr] = np.stack(observations[agent_i, :])
                self.actions[agent_i][self.ptr] = np.stack(actions[agent_i, :])
                self.next_observations[agent_i][self.ptr] = np.stack(next_observations[agent_i, :])
                self.rewards[agent_i][self.ptr] = rewards[agent_i]
                self.dones[agent_i][self.ptr] = dones[agent_i]
            else:
                self.observations[agent_i][self.ptr] = np.stack(torch.zeros([self.state_dim]))
                self.actions[agent_i][self.ptr] = np.stack(torch.zeros([self.action_dim]))
                self.next_observations[agent_i][self.ptr] = np.stack(torch.zeros([self.state_dim]))
                self.rewards[agent_i][self.ptr] = 0.0
                self.dones[agent_i][self.ptr] = 1.0  # True --> 1.0

            self.ptr[agent_i] = (self.ptr[agent_i] + 1) % self.max_size
            self.size[agent_i] = min(self.size[agent_i] + 1, self.max_size)
        # 存入数据的格式：torch.tensor([n_agents, dim])
        # if np.all(np.array(flags)):
        #     for agent_i in range(self.n_agents):
        #         self.observations[agent_i][self.ptr] = np.stack(observations[agent_i, :])
        #         self.actions[agent_i][self.ptr] = np.stack(actions[agent_i, :])
        #         self.next_observations[agent_i][self.ptr] = np.stack(next_observations[agent_i, :])
        #         self.rewards[agent_i][self.ptr] = rewards[agent_i]
        #         self.dones[agent_i][self.ptr] = dones[agent_i]
        #
        #         self.ptr[agent_i] = (self.ptr[agent_i] + 1) % self.max_size
        #         self.size[agent_i] = min(self.size[agent_i] + 1, self.max_size)
        # # 下面这种无法保证时间上的一致性
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
            size = min(size, self.size[agent_i])
        index = np.random.choice(size, size=batch_size)  # Randomly sampling
        batch_obs = torch.tensor([self.observations[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_a = torch.tensor([self.actions[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_r = torch.tensor([self.rewards[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_next_obs = torch.tensor([self.next_observations[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)
        batch_dw = torch.tensor([self.dones[agent_i][index] for agent_i in range(self.n_agents)], dtype=torch.float).to(device)

        return batch_obs, batch_a, batch_r, batch_next_obs, batch_dw

    def __len__(self) -> int:
        return self.size


if __name__ == "__main__":

    starttime = time.time()

    seed = 10
    env_name = "MAStandardEnv"
    env = Env(mode="train")
    policy_name = "NewMATD3"
    state_dim = 15
    action_dim = 3
    max_action = 1.0
    n_agents = 3
    max_timesteps = 1e6
    start_timesteps = 25e3
    eval_freq = 5e3
    expl_noise = 0.1  # Std of Gaussian exploration noise
    batch_size = 64

    policy_noise = 0.2
    noise_clip = 0.5
    update_target_freq = 2     # fixed target

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    file_name = f"{policy_name}_env_{env_name}_seed_{seed}"
    print("---------------------------------------")
    print(f"Policy: {policy_name}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    if not os.path.exists("./eval_reward_train/NewMATD3/"):
        os.makedirs("./eval_reward_train/NewMATD3/")

    if not os.path.exists("./model_train/NewMATD3/"):
        os.makedirs("./model_train/NewMATD3/")

    matd3 = MATD3(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    replay_buffer = MATD3ReplayBuffer(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, max_size=int(max_timesteps))

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

        # env.render()
        # 初始不再是xxx步用于随机探索（即Select action randomly or according to policy）
        # 全都是根据policy
        if t < start_timesteps:
            actions = env.sample_action()
        else:
            actions = matd3.choose_actions(observations=observations)

        # Perform action
        next_observations, rewards, dones, store_flags = env.step(actions)
        # True --> 1.0, False-->0.0(即便到达最后一步，也不算)
        done_bools = [float(dones[i]) if episode_timesteps < env._max_episode_steps else 0 for i in range(n_agents)]

        replay_buffer.add(observations=observations, actions=actions, rewards=rewards,
                          next_observations=next_observations, dones=done_bools, flags=store_flags)
        observations = next_observations
        episode_reward += sum(rewards)

        # Train agent after collecting sufficient data
        # 只训练了一次，DDPG那个代码里面是隔50步训练50次……但标准的TD3其实也是走一步训练一次
        if t >= start_timesteps:
            if t % update_target_freq == 0:
                update_target = True
            else:
                update_target = False
            for i in range(n_agents):
                sample = replay_buffer.sample(batch_size)
                matd3.update(sample, i, update_target)

        # # if t >= start_timesteps and (t + 1) % evaluate_freq == 0:
        # if (t + 1) % evaluate_freq == 0:
        #     evaluate_num += 1
        #     evaluate_reward = eval_policy(matd3)
        #     evaluate_rewards.append(evaluate_reward)
        #     # print("---------------------------------------")
        #     print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
        #     # print("---------------------------------------")
        #     # writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
        #     # Save the rewards
        #     if evaluate_num % 10 == 0:
        #         np.save('./eval_reward_train/MATD3/MATD3_env_{}_seed_{}.npy'.format(env_name, seed),
        #                 np.array(evaluate_rewards))


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
            if (episode_num + 1) % 100 == 0:
                np.save('./eval_reward_train/NewMATD3/train_reward_NewMATD3_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_rewards))
                np.save('./eval_reward_train/NewMATD3/train_ma_reward_NewMATD3_env_{}_seed_{}.npy'.format(env_name, seed),
                        np.array(train_episode_ma_rewards))
            # Reset environment
            env.close()
            env = Env(mode="train")
            observations, dones = env.reset(), [False for i in range(n_agents)]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    env.close()
    np.save('./eval_reward_train/NewMATD3/train_reward_NewMATD3_env_{}_seed_{}.npy'.format(env_name, seed),
            np.array(train_episode_rewards))
    np.save('./eval_reward_train/NewMATD3/train_ma_reward_NewMATD3_env_{}_seed_{}.npy'.format(env_name, seed),
            np.array(train_episode_ma_rewards))
    matd3.save(f"./model_train/NewMATD3/NewMATD3_{env_name}")

    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)
