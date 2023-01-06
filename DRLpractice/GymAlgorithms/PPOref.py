# import argparse
# import datetime
# import time
# import gym
#
# import os
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
# import random
# import torch
# import pandas as pd
#
#
# # 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
# def chinese_font():
#     return None
#
#
# # 中文画图
# def plot_rewards_cn(rewards, cfg, path=None, tag='train'):
#     sns.set()
#     plt.figure()
#     plt.title(u"{}环境下{}算法的学习曲线".format(cfg['env_name'],
#                                        cfg['algo_name']), fontproperties=chinese_font())
#     plt.xlabel(u'回合数', fontproperties=chinese_font())
#     plt.plot(rewards)
#     plt.plot(smooth(rewards))
#     plt.legend(('奖励', '滑动平均奖励',), loc="best", prop=chinese_font())
#     if cfg['save_fig']:
#         plt.savefig(f"{path}/{tag}ing_curve_cn.png")
#     if cfg['show_fig']:
#         plt.show()
#
#
# # 用于平滑曲线，类似于Tensorboard中的smooth
# def smooth(data, weight=0.9):
#     '''
#     Args:
#         data (List):输入数据
#         weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9
#
#     Returns:
#         smoothed (List): 平滑后的数据
#     '''
#     last = data[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in data:
#         smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     return smoothed
#
#
# def plot_rewards(rewards, cfg, path=None, tag='train'):
#     sns.set()
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")
#     plt.xlabel('epsiodes')
#     plt.plot(rewards, label='rewards')
#     plt.plot(smooth(rewards), label='smoothed')
#     plt.legend()
#     if cfg['save_fig']:
#         plt.savefig(f"{path}/{tag}ing_curve.png")
#     if cfg['show_fig']:
#         plt.show()
#
#
# def plot_losses(losses, algo="DQN", save=True, path='./'):
#     sns.set()
#     plt.figure()
#     plt.title("loss curve of {}".format(algo))
#     plt.xlabel('epsiodes')
#     plt.plot(losses, label='rewards')
#     plt.legend()
#     if save:
#         plt.savefig(path + "losses_curve")
#     plt.show()
#
#
# # 保存奖励
# def save_results(res_dic, tag='train', path=None):
#     '''
#     '''
#     Path(path).mkdir(parents=True, exist_ok=True)
#     df = pd.DataFrame(res_dic)
#     df.to_csv(f"{path}/{tag}ing_results.csv", index=None)
#     print('结果已保存: ' + f"{path}/{tag}ing_results.csv")
#
#
# # 创建文件夹
# def make_dir(*paths):
#     for path in paths:
#         Path(path).mkdir(parents=True, exist_ok=True)
#
#
# # 删除目录下所有空文件夹
# def del_empty_dir(*paths):
#     for path in paths:
#         dirs = os.listdir(path)
#         for dir in dirs:
#             if not os.listdir(os.path.join(path, dir)):
#                 os.removedirs(os.path.join(path, dir))
#
#
# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
#
#
# # 保存参数
# def save_args(args, path=None):
#     Path(path).mkdir(parents=True, exist_ok=True)
#     with open(f"{path}/params.json", 'w') as fp:
#         json.dump(args, fp, cls=NpEncoder)
#     print("参数已保存: " + f"{path}/params.json")
#
#
# # 为所有随机因素设置一个统一的种子
# def all_seed(env, seed=520):
#     # 环境种子设置
#     env.seed(seed)
#     # numpy随机数种子设置
#     np.random.seed(seed)
#     # python自带随机数种子设置
#     random.seed(seed)
#     # CPU种子设置
#     torch.manual_seed(seed)
#     # GPU种子设置
#     torch.cuda.manual_seed(seed)
#     # python scripts种子设置
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     # cudnn的配置
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = False
#
#
# class PPOMemory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.batch_size = batch_size
#
#     def sample(self):
#         batch_step = np.arange(0, len(self.states), self.batch_size)
#         indices = np.arange(len(self.states), dtype=np.int64)
#         np.random.shuffle(indices)
#         batches = [indices[i:i + self.batch_size] for i in batch_step]
#         return np.array(self.states), np.array(self.actions), np.array(self.probs), \
#                np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
#
#     def push(self, state, action, probs, vals, reward, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.probs.append(probs)
#         self.vals.append(vals)
#         self.rewards.append(reward)
#         self.dones.append(done)
#
#     def clear(self):
#         self.states = []
#         self.probs = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.vals = []
#
#
# class Actor(torch.nn.Module):
#     def __init__(self, n_states, n_actions,
#                  hidden_dim):
#         super(Actor, self).__init__()
#
#         self.actor = torch.nn.Sequential(
#             torch.nn.Linear(n_states, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, n_actions),
#             torch.nn.Softmax(dim=-1)
#         )
#
#     def forward(self, state):
#         dist = self.actor(state)
#         dist = torch.distributions.categorical(dist)
#         return dist
#
#
# class Critic(torch.nn.Module):
#     def __init__(self, n_states, hidden_dim):
#         super(Critic, self).__init__()
#         self.critic = torch.nn.Sequential(
#             torch.nn.Linear(n_states, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, state):
#         value = self.critic(state)
#         return value
#
#
# class PPO:
#     def __init__(self, n_states, n_actions, cfg):
#         self.gamma = cfg['gamma']
#         self.continuous = cfg['continuous']
#         self.policy_clip = cfg['policy_clip']
#         self.n_epochs = cfg['n_epochs']
#         self.gae_lambda = cfg['gae_lambda']
#         self.device = cfg['device']
#         self.actor = Actor(n_states, n_actions, cfg['hidden_dim']).to(self.device)
#         self.critic = Critic(n_states, cfg['hidden_dim']).to(self.device)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])
#         self.memory = PPOMemory(cfg['batch_size'])
#         self.loss = 0
#
#     def choose_action(self, state):
#         state = np.array([state])  # 先转成数组再转tensor更高效
#         state = torch.tensor(state, dtype=torch.float).to(self.device)
#         dist = self.actor(state)
#         value = self.critic(state)
#         action = dist.sample()
#         probs = torch.squeeze(dist.log_prob(action)).item()
#         if self.continuous:
#             action = torch.tanh(action)
#         else:
#             action = torch.squeeze(action).item()
#         value = torch.squeeze(value).item()
#         return action, probs, value
#
#     def update(self):
#         for _ in range(self.n_epochs):
#             state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.sample()
#             values = vals_arr[:]
#             ### compute advantage ###
#             advantage = np.zeros(len(reward_arr), dtype=np.float32)
#             for t in range(len(reward_arr) - 1):
#                 discount = 1
#                 a_t = 0
#                 for k in range(t, len(reward_arr) - 1):
#                     a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
#                                        (1 - int(dones_arr[k])) - values[k])
#                     discount *= self.gamma * self.gae_lambda
#                 advantage[t] = a_t
#             advantage = torch.tensor(advantage).to(self.device)
#             ### SGD ###
#             values = torch.tensor(values).to(self.device)
#             for batch in batches:
#                 states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
#                 old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
#                 actions = torch.tensor(action_arr[batch]).to(self.device)
#                 dist = self.actor(states)
#                 critic_value = self.critic(states)
#                 critic_value = torch.squeeze(critic_value)
#                 new_probs = dist.log_prob(actions)
#                 prob_ratio = new_probs.exp() / old_probs.exp()
#                 weighted_probs = advantage[batch] * prob_ratio
#                 weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
#                                                      1 + self.policy_clip) * advantage[batch]
#                 actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
#                 returns = advantage[batch] + values[batch]
#                 critic_loss = (returns - critic_value) ** 2
#                 critic_loss = critic_loss.mean()
#                 total_loss = actor_loss + 0.5 * critic_loss
#                 self.loss = total_loss
#                 self.actor_optimizer.zero_grad()
#                 self.critic_optimizer.zero_grad()
#                 total_loss.backward()
#                 self.actor_optimizer.step()
#                 self.critic_optimizer.step()
#         self.memory.clear()
#
#     def save_model(self, path):
#         Path(path).mkdir(parents=True, exist_ok=True)
#         actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
#         critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
#         torch.save(self.actor.state_dict(), actor_checkpoint)
#         torch.save(self.critic.state_dict(), critic_checkpoint)
#
#     def load_model(self, path):
#         actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
#         critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
#         self.actor.load_state_dict(torch.load(actor_checkpoint))
#         self.critic.load_state_dict(torch.load(critic_checkpoint))
#
#
# # 训练函数
# def train(arg_dict, env, agent):
#     # 开始计时
#     startTime = time.time()
#     print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
#     print("开始训练智能体......")
#     rewards = []  # 记录所有回合的奖励
#     ma_rewards = []  # 记录所有回合的滑动平均奖励
#     steps = 0
#     for i_ep in range(arg_dict['train_eps']):
#         state = env.reset()
#         done = False
#         ep_reward = 0
#         while not done:
#             # 画图
#             if arg_dict['train_render']:
#                 env.render()
#             action, prob, val = agent.choose_action(state)
#             state_, reward, done, _ = env.step(action)
#             steps += 1
#             ep_reward += reward
#             agent.memory.push(state, action, prob, val, reward, done)
#             if steps % arg_dict['update_fre'] == 0:
#                 agent.update()
#             state = state_
#         rewards.append(ep_reward)
#         if ma_rewards:
#             ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
#         else:
#             ma_rewards.append(ep_reward)
#         if (i_ep + 1) % 10 == 0:
#             print(f"回合：{i_ep + 1}/{arg_dict['train_eps']}，奖励：{ep_reward:.2f}")
#     print('训练结束 , 用时: ' + str(time.time() - startTime) + " s")
#     # 关闭环境
#     env.close()
#     return {'episodes': range(len(rewards)), 'rewards': rewards}
#
#
# # 测试函数
# def test(arg_dict, env, agent):
#     startTime = time.time()
#     print("开始测试智能体......")
#     print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
#     rewards = []  # 记录所有回合的奖励
#     ma_rewards = []  # 记录所有回合的滑动平均奖励
#     for i_ep in range(arg_dict['test_eps']):
#         state = env.reset()
#         done = False
#         ep_reward = 0
#         while not done:
#             # 画图
#             if arg_dict['test_render']:
#                 env.render()
#             action, prob, val = agent.choose_action(state)
#             state_, reward, done, _ = env.step(action)
#             ep_reward += reward
#             state = state_
#         rewards.append(ep_reward)
#         if ma_rewards:
#             ma_rewards.append(
#                 0.9 * ma_rewards[-1] + 0.1 * ep_reward)
#         else:
#             ma_rewards.append(ep_reward)
#         print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, arg_dict['test_eps'], ep_reward))
#     print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
#     env.close()
#     return {'episodes': range(len(rewards)), 'rewards': rewards}
#
#
# # 创建环境和智能体
# def create_env_agent(arg_dict):
#     # 创建环境
#     env = gym.make(arg_dict['env_name'])
#     # 设置随机种子
#     all_seed(env, seed=arg_dict["seed"])
#     # 获取状态数
#     try:
#         n_states = env.observation_space.n
#     except AttributeError:
#         n_states = env.observation_space.shape[0]
#     # 获取动作数
#     n_actions = env.action_space.n
#     print(f"状态数: {n_states}, 动作数: {n_actions}")
#     # 将状态数和动作数加入算法参数字典
#     arg_dict.update({"n_states": n_states, "n_actions": n_actions})
#     # 实例化智能体对象
#     agent = PPO(n_states, n_actions, arg_dict)
#     # 返回环境，智能体
#     return env, agent
#
#
# if __name__ == '__main__':
#     # # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#     # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     # 获取当前路径
#     curr_path = os.path.dirname(os.path.abspath(__file__))
#     # 获取当前时间
#     curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
#     # 相关参数设置
#     parser = argparse.ArgumentParser(description="hyper parameters")
#     parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
#     parser.add_argument('--env_name', default='CartPole-v0', type=str, help="name of environment")
#     parser.add_argument('--continuous', default=False, type=bool,
#                         help="if PPO is continuous")  # PPO既可适用于连续动作空间，也可以适用于离散动作空间
#     parser.add_argument('--train_eps', default=200, type=int, help="episodes of training")
#     parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
#     parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
#     parser.add_argument('--batch_size', default=5, type=int)  # mini-batch SGD中的批量大小
#     parser.add_argument('--n_epochs', default=4, type=int)
#     parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
#     parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
#     parser.add_argument('--gae_lambda', default=0.95, type=float)
#     parser.add_argument('--policy_clip', default=0.2, type=float)  # PPO-clip中的clip参数，一般是0.1~0.2左右
#     parser.add_argument('--update_fre', default=20, type=int)
#     parser.add_argument('--hidden_dim', default=256, type=int)
#     parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
#     parser.add_argument('--seed', default=520, type=int, help="seed")
#     parser.add_argument('--show_fig', default=False, type=bool, help="if show figure or not")
#     parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
#     parser.add_argument('--train_render', default=False, type=bool,
#                         help="Whether to render the environment during training")
#     parser.add_argument('--test_render', default=True, type=bool,
#                         help="Whether to render the environment during testing")
#     args = parser.parse_args()
#     default_args = {'result_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
#                     'model_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
#                     }
#     # 将参数转化为字典 type(dict)
#     arg_dict = {**vars(args), **default_args}
#     print("算法参数字典:", arg_dict)
#
#     # 创建环境和智能体
#     env, agent = create_env_agent(arg_dict)
#     # 传入算法参数、环境、智能体，然后开始训练
#     res_dic = train(arg_dict, env, agent)
#     print("算法返回结果字典:", res_dic)
#     # 保存相关信息
#     agent.save_model(path=arg_dict['model_path'])
#     save_args(arg_dict, path=arg_dict['result_path'])
#     save_results(res_dic, tag='train', path=arg_dict['result_path'])
#     plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="train")
#
#     # =================================================================================================
#     # 创建新环境和智能体用来测试
#     print("=" * 300)
#     env, agent = create_env_agent(arg_dict)
#     # 加载已保存的智能体
#     agent.load_model(path=arg_dict['model_path'])
#     res_dic = test(arg_dict, env, agent)
#     save_results(res_dic, tag='test', path=arg_dict['result_path'])
#     plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="test")
