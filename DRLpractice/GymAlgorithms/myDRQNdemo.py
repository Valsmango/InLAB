import collections
import itertools
import math
import os
import random
from collections import namedtuple, deque
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count

import time


import torchvision.transforms as T

from torchvision.transforms import InterpolationMode


class CriticNetwork(nn.Module):
    def __init__(self, pic_height, pic_width, output_size):
        # 输入参数 h 是指state图片的height， w 是state图片的width
        # 继承nn.Module：
        super(CriticNetwork, self).__init__()
        # 网络结构：输入state(两张图片之差) - Conv Conv Conv LSTM FC - 输出action动作空间对应的Q值（动作left的Q值、动作right的Q值）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(pic_width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(pic_height)))
        # 一个输出的 conv_width * conv_height * 32 维度的向量表征一个样本
        input_size = conv_width * conv_height * 32
        self.hidden_size = input_size
        '''
        nn.LSTM 参数：
        （1）input_size：表示的是输入的矩阵特征数，或者说是输入的维度；
        （2）hidden_size：隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数；
        （3）num_layers：lstm隐层的层数，默认为1；
        （4）bias：隐层状态是否带bias，默认为true；
        （5）batch_first：True或者False，如果是True，则input为(batch, seq, input_size)，默认值为：False（seq_len, batch, input_size）
              state为 BCHW   B x 3 x 160 x 360，所以要设置为true
        （6）dropout：默认值0，除最后一层，每一层的输出都进行dropout；
        （7）bidirectional：如果设置为True, 则表示双向LSTM，默认为False。
        举个例子：对句子进行LSTM操作，假设有100个句子（sequence）, 每个句子里有7个词，batch_size = 64，embedding_size = 300
        此时，各个参数为：
            input_size = embedding_size = 300
            batch = batch_size = 64
            seq_len = 7
            另外设置hidden_size = 100, num_layers = 1
        代码为：
            import torch
            import torch.nn as nn
            lstm = nn.LSTM(300, 100, 1)
            x = torch.randn(7, 64, 300)
            h0 = torch.randn(1, 64, 100)
            c0 = torch.randn(1, 64, 100)
            output, (hn, cn) = lstm(x, (h0, c0))
        '''
        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)   # 这里用的：nn.LSTM(input_size, hidden_size=input_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def init_hidden_state(self, batch_size, training=None):
        '''
        https://www.cnblogs.com/jiangkejie/p/13246857.html
        '''
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_size]), torch.zeros([1, batch_size, self.hidden_size])
        else:
            return torch.zeros([1, 1, self.hidden_size]), torch.zeros([1, 1, self.hidden_size])

    def forward(self, x, h_0, c_0):
        # x 即为 state，B Seq_len CHW   B x Seq_len x 3 x 160 x 360
        batch, seq_len, channels, height, width = x.size()
        x = x.reshape(batch*seq_len, channels, height, width).to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # nn.LSTM中输入与输出关系为output, (hn, cn) = lstm(input, (h0, c0))
        # https://blog.csdn.net/weixin_43788986/article/details/125441919
        x, (h_t, c_t) = self.lstm(x.reshape(batch, seq_len, -1), (h_0, c_0))
        return self.linear(x), h_t, c_t

    '''
    改进：这里的epsilon改进为自适应的，和 DQN demo 中的一样，随着steps_done增大，epsilon_threshold减小，从0.9减小到0.362（steps_done=200时），到0；
    这里虽然取名叫CriticNetwork，但是因为有了sample_action这个函数，实际上，算是把Cirtic和Actor组成了一个系统的PolicyNetwork
    '''
    def sample_action(self, obs, h, c, epsilon):
        output = self.forward(obs, h, c)

        if random.random() < epsilon:
            return random.randint(0, 1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1], output[2]


class ReplayBuffer:
    def __init__(self, random_update=False, max_epi_num=100, max_epi_len=500, batch_size=1, lookup_step=None):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step
        self.replaybuffer = collections.deque(maxlen=self.max_epi_num)

    def put(self, subreplaybuffer):
        self.replaybuffer.append(subreplaybuffer)   # 每一个subreplaybuffer == 一个episode

    def sample(self):
        sampled_buffer = []
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.replaybuffer, self.batch_size)
            min_step = self.max_epi_len
            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        else:  # 默认情况，Sequential update, 该模式下，batch_size、lookup_step属性失效（或者说batch_size = 1），只是随机找一整个episode
            idx = np.random.randint(0, len(self.replaybuffer))
            sampled_buffer.append(self.replaybuffer[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['done'])  # buffers, sequence_length

    def __len__(self):
        return len(self.replaybuffer)


class SubReplayBuffer:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, s, a, r, next_s, done):
        self.obs.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.next_obs.append(next_s)
        self.done.append(done)

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


class PicUtils:
    """
    图像处理相关
    """
    def __init__(self):
        # 用resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.CUBIC),T.ToTensor()])会报Warning
        self.resize = T.Compose([T.ToPILImage(), # 将tensor格式转换为PIL格式
                    T.Resize(40, interpolation=InterpolationMode.BICUBIC),  # Resize将短边压缩为40，长边按比例变化
                    T.ToTensor()]) # 将PIL格式重新转换为tensor格式

    def get_cart_location(self, screen_width):
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen)  # 用unsqueeze函数增加一个维度，即BCHW，如果gym得图像是400x600x3，则输出为3x160x360的图像


def optimize_model(Q_net=None, Q_target_net=None, replay_buffer=None, batch_size=1, gamma=0.99, channel=3, height=None, width=None):
    samples, seq_len = replay_buffer.sample()

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

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(observations.reshape(batch_size, seq_len, channel, height, width)).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size, seq_len, -1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size, seq_len, -1)).to(device)
    next_observations = torch.FloatTensor(next_observations.reshape(batch_size, seq_len, channel, height, width)).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size, seq_len, -1)).to(device)

    h_target, c_target = Q_target_net.init_hidden_state(batch_size=batch_size, training=True)

    q_target, _, _ = Q_target_net(next_observations, h_target.to(device), c_target.to(device))

    q_target_max = q_target.max(2)[0].view(batch_size, seq_len, -1).detach()
    targets = rewards + gamma * q_target_max * dones

    h, c = Q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, _, _ = Q_net(observations, h.to(device), c.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    t0 = time.time()

    # Env parameters
    model_name = "myDRQN"
    env_name = "CartPole-v0"
    seed = 10
    save_model_flag = True

    # Set gym environment
    env = gym.make(env_name)

    # Set seeds
    env.seed(seed)
    # env.action_space.seed(seed)     # 这个是TD3中提到的seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save results
    file_name = f"{model_name}_{env_name}_{seed}"
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model_flag and not os.path.exists("./models"):
        os.makedirs("./models")

    # model parameters
    # 适当设置batch size，过小的经验池容量和batchsize导致收敛到局部最优，结果呈现震荡形式
    learning_rate = 1e-3
    batch_size = 16
    tau = 1e-2
    max_steps = 2000
    max_episodes = 600
    min_epi_num = 64  # Start moment to train the Q network

    # Initiate the network and set the optimizer
    env.reset()
    picUtil = PicUtils()
    init_screen = picUtil.get_screen()
    _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n
    Q_net = CriticNetwork(screen_height, screen_width, n_actions).to(device)
    Q_target_net = CriticNetwork(screen_height, screen_width, n_actions).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=learning_rate)

    # Initiate the ReplayBuffer
    replay_buffer = ReplayBuffer(max_epi_num=200, max_epi_len=600,
                                 random_update=True, batch_size=batch_size, lookup_step=20)

    # other parameters
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    epsilon = eps_start
    gamma = 0.99
    TARGET_UPDATE = 4

    # output the reward
    print_per_iter = 20
    score = 0
    score_sum = 0.0

    # 开始训练
    for i in range(max_episodes):
        # Initialize the environment and state
        env.reset()
        state = picUtil.get_screen()
        h, c = Q_net.init_hidden_state(batch_size=batch_size, training=False)

        # 创建存储当前episode的临时buffer
        sub_replay_buffer = SubReplayBuffer()
        
        for t in range(max_steps):
            # Select and perform an action
            action, h, c = Q_net.sample_action(state.unsqueeze(0).unsqueeze(0).to(device),  # state 为 B seq CHW     B x seq x 3 x 160 x 360
                                               h=h.to(device),
                                               c=c.to(device),
                                               epsilon=epsilon)

            _, reward, done, _ = env.step(action)
            # reward是一个float格式的数值
            score += reward
            score_sum += reward

            next_state = picUtil.get_screen()
            done_mask = 0.0 if done else 1.0
            sub_replay_buffer.put(s=np.array(state), a=action, next_s=np.array(next_state), r=reward, done=done_mask)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if len(replay_buffer) >= min_epi_num:
                optimize_model(Q_net=Q_net, Q_target_net=Q_target_net, replay_buffer=replay_buffer, batch_size=batch_size, gamma=gamma, height=screen_height, width=screen_width)

                if (t + 1) % TARGET_UPDATE == 0:
                    # Q_target_net.load_state_dict(Q_net.state_dict()) # naive update
                    for target_param, local_param in zip(Q_target_net.parameters(), Q_net.parameters()):  # <- soft update
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing
        replay_buffer.put(sub_replay_buffer)
        score = 0

        # 隔一段时间，输出一次训练的结果
        if i % print_per_iter == 0 and i != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                i, score_sum / print_per_iter, len(replay_buffer), epsilon * 100))
            score_sum = 0.0

    # save the model
    if save_model_flag:
        torch.save(Q_net.state_dict(), f"./models/{file_name}")

    t1 = time.time()
    print(f"运行时间为：{t1-t0:.2f}s")

'''
https://zhuanlan.zhihu.com/p/524650878
'''

'''
n_episode :20, score : 9.9, n_buffer : 21, eps : 9.0%
n_episode :40, score : 18.2, n_buffer : 41, eps : 8.1%
n_episode :60, score : 32.1, n_buffer : 61, eps : 7.4%
n_episode :80, score : 44.3, n_buffer : 81, eps : 6.7%
n_episode :100, score : 50.6, n_buffer : 100, eps : 6.0%
n_episode :120, score : 61.6, n_buffer : 100, eps : 5.5%
n_episode :140, score : 96.8, n_buffer : 100, eps : 4.9%
n_episode :160, score : 53.5, n_buffer : 100, eps : 4.5%
n_episode :180, score : 36.9, n_buffer : 100, eps : 4.0%
n_episode :200, score : 56.8, n_buffer : 100, eps : 3.7%
n_episode :220, score : 58.9, n_buffer : 100, eps : 3.3%
n_episode :240, score : 73.5, n_buffer : 100, eps : 3.0%
n_episode :260, score : 100.2, n_buffer : 100, eps : 2.7%
n_episode :280, score : 120.0, n_buffer : 100, eps : 2.4%
n_episode :300, score : 58.6, n_buffer : 100, eps : 2.2%
n_episode :320, score : 61.5, n_buffer : 100, eps : 2.0%
n_episode :340, score : 66.2, n_buffer : 100, eps : 1.8%
n_episode :360, score : 51.5, n_buffer : 100, eps : 1.6%
n_episode :380, score : 69.1, n_buffer : 100, eps : 1.5%
n_episode :400, score : 73.5, n_buffer : 100, eps : 1.3%
n_episode :420, score : 67.3, n_buffer : 100, eps : 1.2%
n_episode :440, score : 62.7, n_buffer : 100, eps : 1.1%
n_episode :460, score : 60.5, n_buffer : 100, eps : 1.0%
n_episode :480, score : 79.8, n_buffer : 100, eps : 0.9%
n_episode :500, score : 78.0, n_buffer : 100, eps : 0.8%
n_episode :520, score : 79.5, n_buffer : 100, eps : 0.7%
n_episode :540, score : 78.5, n_buffer : 100, eps : 0.7%
n_episode :560, score : 79.0, n_buffer : 100, eps : 0.6%
n_episode :580, score : 60.1, n_buffer : 100, eps : 0.5%
n_episode :600, score : 78.7, n_buffer : 100, eps : 0.5%
n_episode :620, score : 62.9, n_buffer : 100, eps : 0.4%
n_episode :640, score : 70.3, n_buffer : 100, eps : 0.4%
n_episode :660, score : 41.5, n_buffer : 100, eps : 0.4%
n_episode :680, score : 69.5, n_buffer : 100, eps : 0.3%
n_episode :700, score : 55.1, n_buffer : 100, eps : 0.3%
n_episode :720, score : 52.8, n_buffer : 100, eps : 0.3%
n_episode :740, score : 68.8, n_buffer : 100, eps : 0.2%
n_episode :760, score : 56.3, n_buffer : 100, eps : 0.2%
n_episode :780, score : 45.3, n_buffer : 100, eps : 0.2%
n_episode :800, score : 68.7, n_buffer : 100, eps : 0.2%
n_episode :820, score : 54.5, n_buffer : 100, eps : 0.2%
n_episode :840, score : 82.5, n_buffer : 100, eps : 0.1%
n_episode :860, score : 74.7, n_buffer : 100, eps : 0.1%
n_episode :880, score : 75.8, n_buffer : 100, eps : 0.1%
n_episode :900, score : 69.8, n_buffer : 100, eps : 0.1%
n_episode :920, score : 46.5, n_buffer : 100, eps : 0.1%
n_episode :940, score : 60.4, n_buffer : 100, eps : 0.1%
n_episode :960, score : 53.9, n_buffer : 100, eps : 0.1%
n_episode :980, score : 32.2, n_buffer : 100, eps : 0.1%

进程已结束，退出代码 0

n_episode :20, score : 9.9, n_buffer : 21, eps : 9.0%
n_episode :40, score : 9.7, n_buffer : 41, eps : 8.1%
n_episode :60, score : 9.3, n_buffer : 61, eps : 7.4%
n_episode :80, score : 12.4, n_buffer : 81, eps : 6.7%
n_episode :100, score : 24.8, n_buffer : 101, eps : 6.0%
n_episode :120, score : 26.2, n_buffer : 121, eps : 5.5%
n_episode :140, score : 34.3, n_buffer : 141, eps : 4.9%
n_episode :160, score : 37.5, n_buffer : 161, eps : 4.5%
n_episode :180, score : 80.4, n_buffer : 181, eps : 4.0%
n_episode :200, score : 82.8, n_buffer : 200, eps : 3.7%
n_episode :220, score : 57.8, n_buffer : 200, eps : 3.3%
n_episode :240, score : 78.2, n_buffer : 200, eps : 3.0%
n_episode :260, score : 100.7, n_buffer : 200, eps : 2.7%
n_episode :280, score : 84.7, n_buffer : 200, eps : 2.4%
n_episode :300, score : 117.7, n_buffer : 200, eps : 2.2%
n_episode :320, score : 98.9, n_buffer : 200, eps : 2.0%
n_episode :340, score : 99.3, n_buffer : 200, eps : 1.8%
n_episode :360, score : 94.8, n_buffer : 200, eps : 1.6%
n_episode :380, score : 80.8, n_buffer : 200, eps : 1.5%
n_episode :400, score : 92.5, n_buffer : 200, eps : 1.3%
n_episode :420, score : 77.5, n_buffer : 200, eps : 1.2%
n_episode :440, score : 57.2, n_buffer : 200, eps : 1.1%
n_episode :460, score : 49.0, n_buffer : 200, eps : 1.0%
n_episode :480, score : 65.3, n_buffer : 200, eps : 0.9%
n_episode :500, score : 82.0, n_buffer : 200, eps : 0.8%
n_episode :520, score : 57.8, n_buffer : 200, eps : 0.7%
n_episode :540, score : 76.7, n_buffer : 200, eps : 0.7%
n_episode :560, score : 59.6, n_buffer : 200, eps : 0.6%
n_episode :580, score : 59.8, n_buffer : 200, eps : 0.5%
运行时间为：2655.29s

进程已结束，退出代码 0
'''
