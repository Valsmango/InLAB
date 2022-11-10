import math
import os
import random
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count


class CriticNetwork(nn.module):
    def __init__(self, h, w):
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
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        input_size = convw * convh * 32

        self.lstm = nn.LSTM(input_size, input_size, batch_first=True)
        self.linear = nn.Linear(input_size, 2)

    def forward(self, x, h, c):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, (_, _) = self.lstm(x, h, c)
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
    save_model_flag = True

    # Set gym environment
    env = gym.make(env_name)

    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)     # 这个是TD3中提到的seed
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
    batch_sz = 16
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 2000

    # Initiate the network and set the optimizer
    Q_net = CriticNetwork()
    Q_target_net = CriticNetwork()
    optimizer = optim.Adam(Q_net.parameters(), lr=learning_rate)

    # training parameters
    max_episodes = 1000
    h =
    c =



    # 开始训练
    for i in range(max_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = Q_net.sample_action(state, h=h, c=c ,epsilon=)

            _, reward, done, _ = env.step(action.item())
            # reward是一个float格式的数值，转换为tensor
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # 更新eps
            epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # save the model
    if save_model_flag:
        torch.save(Q_net.state_dict(), file_name)


'''
https://zhuanlan.zhihu.com/p/524650878
'''
