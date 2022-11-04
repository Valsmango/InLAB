import random
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CriticNetwork(nn.module):
    def __init__(self):
        # 继承nn.Module：
        super(CriticNetwork, self).__init__()
        # 网络结构：输入state(两张图片之差) - Conv Conv Conv LSTM FC - 输出action动作空间对应的Q值（动作left的Q值、动作right的Q值）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.lstm = nn.LSTM()
        self.linear = nn.Linear()
        # self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x, h, c):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.lstm(x, h, c)
        return self.linear(x.view(x.size(0), -1))

    '''
    改进：这里的epsilon可以改进为自适应的，
    例如，和 DQN demo 中的一样，随着steps_done增大，epsilon_threshold减小，从0.9减小到0.362（steps_done=200时），到0；
    '''
    def sample_action(self, obs, h, c, epsilon):
        output = self.forward(obs, h, c)

        if random.random() < epsilon:
            return random.randint(0, 1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1], output[2]


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

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


if __name__ == "__main__":

    # Env parameters
    model_name = "myDRQN"
    env_name = "CartPole-v0"
    seed = 0
    exp_num = 'SEED' + '_' + str(seed)

    # Set gym environment
    env = gym.make(env_name)
    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    env.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")