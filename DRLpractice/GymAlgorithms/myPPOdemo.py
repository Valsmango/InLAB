import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter

import DRLpractice.GymAlgorithms.plotUtils.PlotUtils as pltutil
import time

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, batch_size):
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []
        self.batch_size = batch_size

    def put(self, s, a, r, s_next, done):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.next_state.append(s_next)
        self.done.append(done)

    def sample(self):
        '''
        ������������ֻ��Ҫ��ǰ����batch���أ�
        '''
        batch_step = np.arange(0, len(self.done), self.batch_size)  # np.arange(start = 0, stop = len, step = batch_sz) ���� ��0�� batch_sz, batch_sz*2, ... < stop��
        indicies = np.arange(len(self.done), dtype=np.int64)  # np.arange(start = 0, stop = len, step = 1)
        np.random.shuffle(indicies)     # ��indiciesŪ������
        batches = [indicies[i:i+self.batch_size] for i in batch_step]
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_done = []
        for batch in batches:
            batch_state.append(np.array(self.state)[batch])
            batch_action.append(np.array(self.action)[batch])
            batch_reward.append(np.array(self.reward)[batch])
            batch_next_state.append(np.array(self.next_state)[batch])
            batch_done.append(np.array(self.done)[batch])
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def clear(self):
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []

    def __len__(self) -> int:
        return len(self.done)


class CriticNetwork(nn.Module):
    '''
    Critic����Ϊ��state
    ���Ϊ�� value(state)�Ĺ���ֵ
    '''
    def __init__(self, state_space, output_size=1, hidden_space=64):
        # �̳�nn.Module��
        super(CriticNetwork, self).__init__()

        self.hidden_space = hidden_space
        self.state_space = state_space
        self.output_size = output_size

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.output_size)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)


class DiscreteActorNetwork(nn.Module):
    '''
    Actror����Ϊ��state
    ���Ϊ�� action���ʷֲ�

        ����Բ��ԣ��൱��ѵ�������е�̽��exploration����Ӧ�Ĳ������ǲ���
        ǿ��ѧϰ�������ֳ���������Բ��ԣ�
            �������categorical policy��������ɢ�����ռ�����
            ������Կ��Կ�����ɢ�����ռ�ķ����� ���� �����ǹ۲⣬����һЩ������㣬
            ���ÿ��������logits�������softmaxת��Ϊÿһ�������ĸ���probability
            ����ÿһ�������ĸ��ʣ�����ʹ��Pytorch�е�һЩ�����������в���������Categorical distributions in PyTorch��torch.multinomial
    '''
    def __init__(self, state_space, action_space, hidden_space=64):
        super(DiscreteActorNetwork, self).__init__()

        self.hidden_space = hidden_space
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.softmax(self.Linear3(x), dim=-1)
        x = Categorical(x)
        return x


class DiscretePPO:
    def __init__(self, batch_size, state_space, action_space, clip,
                 actor_lr, critic_lr, epochs, gamma, gae_lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.clip = clip

        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.batch_size)
        self.actor = DiscreteActorNetwork(state_space=self.state_space, action_space=self.action_space)
        self.critic = CriticNetwork(state_space=self.state_space)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.critic_lr)

        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def sample_action(self, state):
        '''
        state: ����ʱ�ĸ�ʽӦ��Ϊtodevice�˵�tensor
        Ϊʲô���� epsilon greedy ��������������������������������
        '''
        state = np.array([state])  # ��ת��������תtensor����Ч
        state = torch.tensor(state, dtype=torch.float).to(device)
        distribution = self.actor(state)
        action = distribution.sample()
        probs = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()

        value = self.critic(state)
        value = torch.squeeze(value).item()

        return action, probs, value

    def update(self):
        '''
        ÿ��һ��ʱ�����һ��
        '''
        for _ in range(self.epochs):
            state, action, reward, next_state, done = self.memory.sample()
            state = torch.tensor(state, dtype=torch.float).to(device)



            # GAE
            advantage = np.zeros(len(reward), dtype=np.float32)
            for i in range(len(reward) - 1):
                discount = 1
                a_i = 0
                for k in range(i, len(reward) - 1):
                    # ����Ҫע�⣬��done��ʱ��gamma * v��next_state�� == 0
                    a_i += discount * (reward[k] + self.gamma * self.critic(next_state) * int(done[k]) - self.critic(state))
                    discount *= self.gamma * self.gae_lambda
                advantage[i] = a_i
            advantage = torch.tensor(advantage).to(device)
            value = torch.tensor(value).to(device)
            for batch in batches:
                batch_state = torch.tensor(np.array(state)[batch], dtype=torch.float).to(device)

                batch_action = torch.tensor(np.array(action)[batch]).to(device)
                # ����batch_new_prob
                distribution = self.actor(batch_state)
                batch_new_prob = distribution.log_prob(batch_action)    # ????????????
                # ����actor��loss�������ĵĸ��¹�ʽ
                prob_ratio = batch_new_prob.exp() / batch_old_prob.exp()
                weighted_prob = prob_ratio * advantage[batch]
                weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage[batch]
                actor_loss = -torch.min(weighted_prob, weighted_clipped_prob).mean()
                # ����actor����
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # ����critic��loss
                batch_critic_value = torch.squeeze(self.critic(batch_state))
                batch_real_value = advantage[batch] + value[batch]
                critic_loss = F.mse_loss(batch_real_value,
                                         batch_critic_value)  # critic_loss = (batch_real_value - batch_critic_value) ** 2    # critic_loss = critic_loss.mean()
                # ����critic����
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        self.memory.clear()

    def save_model(self):
        pass

    def load_model(self):
        pass

    def to_device(self):
        self.actor.to(device)
        self.critic.to(device)


if __name__ == "__main__":
    start_time = time.time()

    # Env parameters
    model_name = "PPO"
    env_name = "CartPole-v0"
    seed = 10
    save_model_flag = False
    # writer = SummaryWriter("./runs/tensorboard/ppo")

    # Set gym environment
    env = gym.make(env_name)
    n_actions = env.action_space.n
    try:
        n_states = env.observation_space.n
    except AttributeError:
        n_states = env.observation_space.shape[0]

    # Set seeds
    env.seed(seed)
    # env.action_space.seed(seed)     # �����TD3���ᵽ��seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # save results
    file_name = f"{model_name}_{env_name}_{seed}"
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model_flag and not os.path.exists("./models"):
        os.makedirs("./models")

    # model parameters
    # �ʵ�����batch size����С�ľ����������batchsize�����������ֲ����ţ������������ʽ
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-3
    batch_size = 8
    max_episodes = 200
    clip = 0.2
    gamma = 0.99
    gae_lambda = 0.92
    step_count = 0
    epochs = 10
    update_frequent = 20

    # Initiate the network and set the optimizer
    env.reset()
    model = DiscretePPO(batch_size=batch_size, state_space=n_states, action_space=n_actions, clip=clip,
                        actor_lr=actor_learning_rate, critic_lr=critic_learning_rate,
                        epochs=epochs, gamma=gamma, gae_lambda=gae_lambda)
    model.to_device()

    # output the reward
    rewards = []
    ma_rewards = []
    print_per_iter = 20     # ÿ��1����Ϸ����һ�ν�����
    score = 0
    score_sum = 0.0

    # ��ʼѵ��
    for i in range(max_episodes):
        # Initialize the environment and state
        state = env.reset()
        done = None
        while not done:
            # ��Ⱦ
            env.render()

            # Select and perform an action
            action, prob, val = model.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[::2]    # CartPole-v0
            # reward��һ��float��ʽ����ֵ
            score += reward
            score_sum += reward
            done_mask = 0.0 if done else 1.0
            model.memory.put(s=state, a=action, r=reward, done=done_mask, p=prob, v=val)

            # Move to the next state
            state = next_state
            step_count += 1

            # Perform one step of the optimization
            if step_count % update_frequent == 0:
                model.update()

        rewards.append(score)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * score)
        else:
            ma_rewards.append(score)
        # ��һ��ʱ�䣬���һ��ѵ���Ľ��
        if i % print_per_iter == 0 and i != 0:
            print("n_episode :{}, score : {:.1f}".format(i, score))
        # writer.add_scalar("rewards", score, i + 1)
        score = 0
    # save the model
    if save_model_flag:
        pass

    end_time = time.time()
    print("-----------------------------------------")
    print(f"����ʱ��Ϊ��{end_time - start_time:.2f}s")
    print("-----------------------------------------")
    pltutil.plot_certain_training_rewards(rewards, "PPO", "PPO-CartPole rewards")
