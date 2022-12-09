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
        # x 即为 state，BCHW   B x 3 x 160 x 360
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # nn.LSTM中输入与输出关系为output, (hn, cn) = lstm(input, (h0, c0))
        # https://blog.csdn.net/weixin_43788986/article/details/125441919
        x, (h_t, c_t) = self.lstm(x.view(x.size(0), -1).unsqueeze(0), (h_0, c_0))
        return self.linear(x.view(x.size(0), -1)), h_t, c_t

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

        return sampled_buffer, len(sampled_buffer[0])  # buffers, sequence_length

    def __len__(self):
        return len(self.replaybuffer)


class SubReplayBuffer:
    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                        ('obs', 'action', 'next_obs', 'reward', 'done'))
        self.subreplaybuffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.subreplaybuffer.append(self.Transition(*args))

    def sample(self, random_update=False, idx=None, lookup_step=None):
        if random_update is False:
            return self.subreplaybuffer
        else:
            return collections.deque(itertools.islice(self.subreplaybuffer, idx, idx + lookup_step))

    def __len__(self):
        return len(self.subreplaybuffer)


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
        return self.resize(screen).unsqueeze(0)  # 用unsqueeze函数增加一个维度，即BCHW，如果gym得图像是400x600x3，则输出为3x160x360的图像


def optimize_model(Q_net, Q_target_net, replay_buffer, batch_size, gamma):
    # 1, sample，选择默认的sequential update
    seq_len = 0
    while(seq_len <= batch_size):
        samples, seq_len = replay_buffer.sample()
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    # sample = samples[0]
    for i in range(batch_size):
        # 每一个sample都是多个Transition（一步）
        # observations.append(sample[j].__getattribute__("obs"))
        # actions.append(sample[j][1])
        # rewards.append(sample[j][2])
        # next_observations.append(sample[j][3])
        # dones.append(sample[j][4])
        for j in range(seq_len):
            observations.append(samples[i][j].__getattribute__("obs"))
            actions.append(samples[i][j].__getattribute__("action"))
            rewards.append(samples[i][j].__getattribute__("reward"))
            next_observations.append(samples[i][j].__getattribute__("next_obs"))
            dones.append(samples[i][j].__getattribute__("done"))

    observations = torch.stack(list(observations)).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.stack(list(rewards)).to(device)
    next_observations = torch.stack(list(next_observations)).to(device)
    dones = torch.stack(list(dones)).to(device)


    #



    # h_target, c_target = Q_target_net.init_hidden_state(batch_size=batch_size, training=True)
    h_target, c_target = Q_target_net.init_hidden_state(batch_size=1, training=True)

    q_target, _, _ = Q_target_net(next_observations, h_target.to(device), c_target.to(device))

    q_target_max = q_target.max(2)[0].detach()
    expected_q_a = rewards + gamma * q_target_max * dones

    # h, c = Q_net.init_hidden_state(batch_size=batch_size, training=True)
    h, c = Q_net.init_hidden_state(batch_size=1, training=True)
    q_out, _, _ = Q_net(observations, h.to(device), c.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, expected_q_a)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    for param in Q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == "__main__":
    # Env parameters
    model_name = "myDRQN"
    env_name = "CartPole-v0"
    seed = 0
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
    learning_rate = 1e-3
    batch_size = 16
    # eps_start = 0.1
    # eps_end = 0.001
    # eps_decay = 0.995
    tau = 1e-2
    max_step = 2000
    max_episodes = 1000
    min_epi_num = 20  # Start moment to train the Q network

    # Initiate the network and set the optimizer
    env.reset()
    picUtil = PicUtils()
    init_screen = picUtil.get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n
    Q_net = CriticNetwork(screen_height, screen_width, n_actions).to(device)
    Q_target_net = CriticNetwork(screen_height, screen_width, n_actions).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=learning_rate)

    # Initiate the ReplayBuffer
    replay_buffer = ReplayBuffer(max_epi_num=max_episodes)  # 初始测试采用sequential update
    # replay_buffer = ReplayBuffer(max_epi_num=max_episodes, random_update=True, batch_size=batch_size, lookup_step=20)

    # other parameters
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200  # 越小，eps_threshold下降得越快
    TARGET_UPDATE = 5
    steps_done = 0
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    gamma = 0.99

    # output the reward
    print_per_iter = 20
    score = 0
    score_sum = 0.0

    # 开始训练
    for i in range(max_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = picUtil.get_screen()
        current_screen = picUtil.get_screen()
        state = current_screen - last_screen        # state 为 BCHW     B x 3 x 160 x 360
        h, c = Q_net.init_hidden_state(batch_size=batch_size, training=False)

        # 创建存储当前episode的临时buffer
        sub_replay_buffer = SubReplayBuffer(max_step)
        
        for t in count():
            # Select and perform an action
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            action, h, c = Q_net.sample_action(state.to(device),
                                               h=h.to(device),
                                               c=c.to(device),
                                               epsilon=eps_threshold)

            _, reward, done, _ = env.step(action) # 直接action和action.item()有什么区别呢？
            # reward是一个float格式的数值，转换为tensor
            reward = torch.tensor([reward], device=device)
            score += reward
            score_sum += reward

            # Observe new state
            last_screen = current_screen
            current_screen = picUtil.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            done_mask = 0.0 if done else 1.0
            # Store the transition in sub_replay_buffer
            sub_replay_buffer.push(state, action, next_state, reward, done_mask)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if len(replay_buffer) >= min_epi_num:
                optimize_model(Q_net, Q_target_net, replay_buffer, batch_size, gamma)

                if (t + 1) % TARGET_UPDATE == 0:
                    # Q_target_net.load_state_dict(Q_net.state_dict()) # naive update
                    for target_param, local_param in zip(Q_target_net.parameters(), Q_net.parameters()):  # <- soft update
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

            # # 更新eps, 因为采用了DQN的渐进，放弃了这里的linear annealing
            # epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing

            if done:
                break

        replay_buffer.put(sub_replay_buffer)
        score = 0

        # 隔一段时间，输出一次训练的结果
        if i % print_per_iter == 0 and i != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                i, score_sum / print_per_iter, len(replay_buffer), eps_threshold * 100))
            score_sum = 0.0

    # save the model
    if save_model_flag:
        torch.save(Q_net.state_dict(), file_name)


'''
https://zhuanlan.zhihu.com/p/524650878
'''
