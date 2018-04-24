import argparse
import gym
import gym_traffic
import numpy as np
from itertools import count
from collections import namedtuple, deque
from skimage.transform import resize
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from logger import Logger
import pdb
import os
# import cv2


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('Traffic-Multi-cli-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorNetwork(nn.Module):
    def __init__(self, num_agents):
        super(ActorNetwork, self).__init__()
        # self.affine1 = nn.Linear(4, 128)
        # self.action_head = nn.Linear(128, 2)
        self.conv1 = nn.Conv2d(2, 32, (8,8), (4,4))
        self.conv2 = nn.Conv2d(32, 64, (4,4), (2,2))
        self.conv3 = nn.Conv2d(64, 64, (3,3), (1,1))
        # self.conv4 = nn.Conv2d(64, 512, (7,7), (1,1))
        self.rnn1 = nn.LSTM(input_size=3136, hidden_size=512, num_layers=2)
        self.action1 = nn.Linear(512, 128)
        self.action2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, 3)
        self.num_agents = num_agents
        self.agent_ids = [num for num in range(num_agents)]
        self.live_agents = list(self.agent_ids)
        self.saved_actions_dict = {num:[] for num in range(self.num_agents)}
        self.rewards_dict = {num:[] for num in range(self.num_agents)}
        self.h_n_dict = {num:None for num in range(self.num_agents)}
        self.c_n_dict = {num:None for num in range(self.num_agents)}
        # self.h_n = torch.randn(2, 3, 20)
        # self.c_n = torch.randn(2, 3, 20)
        self.time_steps = 1
        self.reset = False

    def forward(self, x, agent_id):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv4 = F.relu(self.conv4(conv3))
        # print(conv3.size())
        # print(conv4.size())
        flatten = conv3.view(self.time_steps,1,-1)
        if self.reset or not self.h_n_dict[agent_id] or not self.c_n_dict[agent_id]:
            lstm1, (self.h_n_dict[agent_id], self.c_n_dict[agent_id]) = self.rnn1(flatten)
        else:
            lstm1, (self.h_n_dict[agent_id], self.c_n_dict[agent_id]) = self.rnn1(flatten, (self.h_n_dict[agent_id], self.c_n_dict[agent_id]))
            self.reset = False
        # print(flatten.size())
        flatten = lstm1.view(1, -1)
        action = F.relu(self.action1(flatten))
        action = F.relu(self.action2(action))
        action_scores = self.action_head(action)
        # print(action_scores)
        # value = F.relu(self.value1(flatten))
        # value = F.relu(self.value2(value))
        # state_values = self.value_head(value)
        # print(state_values)
        return F.softmax(action_scores, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, num_agents):
        super(CriticNetwork, self).__init__()
        # self.affine1 = nn.Linear(4, 128)
        # self.action_head = nn.Linear(128, 2)
        self.conv1 = nn.Conv2d(2*num_agents, 32, (8,8), (4,4))
        self.conv2 = nn.Conv2d(32, 64, (4,4), (2,2))
        self.conv3 = nn.Conv2d(64, 64, (3,3), (1,1))
        # self.conv4 = nn.Conv2d(64, 512, (7,7), (1,1))
        self.rnn1 = nn.LSTM(input_size=3136, hidden_size=512, num_layers=2)
        # self.action1 = nn.Linear(512, 128)
        # self.action2 = nn.Linear(128, 64)
        # self.action_head = nn.Linear(64, 3)
        self.value1 = nn.Linear(512, 128)
        self.value2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        # self.num_agents = num_agents
        # self.agent_ids = [num for num in range(num_agents)]
        # self.live_agents = list(self.agent_ids)
        # self.saved_actions_dict = {num:[] for num in range(self.num_agents)}
        # self.rewards_dict = {num:[] for num in range(self.num_agents)}
        # self.h_n_dict = {num:None for num in range(self.num_agents)}
        # self.c_n_dict = {num:None for num in range(self.num_agents)}
        self.h_n = None
        self.c_n = None
        # self.saved_actions = []
        # self.rewards = []
        self.reset = False
        self.time_steps = 1

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv4 = F.relu(self.conv4(conv3))
        # print(conv3.size())
        # print(conv4.size())
        flatten = conv3.view(self.time_steps,1,-1)
        if self.reset or not self.h_n or not self.c_n:
            lstm1, (self.h_n, self.c_n) = self.rnn1(flatten)
        else:
            lstm1, (self.h_n, self.c_n) = self.rnn1(flatten, (self.h_n, self.c_n))
            self.reset = False
        # print(flatten.size())
        flatten = lstm1.view(1, -1)
        # action = F.relu(self.action1(flatten))
        # action = F.relu(self.action2(action))
        # action_scores = self.action_head(action)
        # print(action_scores)
        value = F.relu(self.value1(flatten))
        value = F.relu(self.value2(value))
        state_values = self.value_head(value)
        # print(state_values)
        return state_values

actor_model = ActorNetwork(num_agents=4)
print(actor_model)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.0001)

critic_model = CriticNetwork(num_agents=4)
print(critic_model)
critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.0001)

def select_action(state, agent_id, state_value):
    # if agent_id == 0:
    #     # print(state.shape)
    #     cv2.imshow('image',state[0,:,:] + state[1,:,:])
    #     cv2.waitKey(1)
    state = torch.from_numpy(state).float()
    probs = actor_model(Variable(state).unsqueeze(0), agent_id)
    m = Categorical(probs)
    action = m.sample()
    actor_model.saved_actions_dict[agent_id].append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]


def finish_episode():
    saved_actions_dict = actor_model.saved_actions_dict
    rewards_dict = {num:[] for num in range(actor_model.num_agents)}
    policy_losses = []
    value_losses = []

    for agent_id in range(actor_model.num_agents):
        R = 0
        for r in actor_model.rewards_dict[agent_id][::-1]:
            R = r + args.gamma * R
            rewards_dict[agent_id].insert(0, R)
        rewards_dict[agent_id] = torch.Tensor(rewards_dict[agent_id])
        rewards_dict[agent_id] = (rewards_dict[agent_id] - rewards_dict[agent_id].mean()) / (rewards_dict[agent_id].std() + np.finfo(np.float32).eps)
        # pdb.set_trace()
        for (log_prob, value), r in zip(saved_actions_dict[agent_id], rewards_dict[agent_id]):
            reward = r - value.data[0]
            policy_losses.append(-log_prob * Variable(reward))
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    # print(value_losses)
    # print(policy_losses)
    actor_loss = torch.stack(policy_losses).sum()/float(actor_model.num_agents)
    actor_loss.backward()

    critic_loss = torch.stack(value_losses).sum()/float(actor_model.num_agents)
    critic_loss.backward()

    actor_optimizer.step()
    critic_optimizer.step()

    actor_model.rewards_dict.clear()
    actor_model.saved_actions_dict.clear()
    return loss

def save_checkpoint(state, is_best, folder='model/multiple_central_critic', filename='checkpoint.pth.tar'):
    print("save checkpoint")

    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(filename, os.path.join(folder, 'model_best.pth.tar'))

def load_checkpoint(model, filename='model/multiple_recurrent/checkpoint.pth.tar'):
    if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            episode = checkpoint['episode']
            max_reward = checkpoint['best_reward']
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['episode']))
    return model

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def main():
    queue = deque([], maxlen=10)
    log_path = 'log'
    from datetime import datetime
    now = datetime.now()
    log_path = "log/multiple_recurrent"
    logger = Logger(log_path,  now.strftime("%Y%m%d-%H%M%S"))
    is_best = False
    running_reward = 0
    max_reward = -float('inf')
    BPTT = 30
    for i_episode in count(1):
        states = env.reset()
        actor_model.live_agents = list(actor_model.agent_ids)
        actor_model.saved_actions_dict = {num:[] for num in range(actor_model.num_agents)}
        actor_model.rewards_dict = {num:[] for num in range(actor_model.num_agents)}
        actor_model.reset = True
        critic_model.reset = True
        for t in range(10000):  # Don't infinite loop while learning
            if t % BPTT == 0 and t != 0:
                for agent_id in actor_model.live_agents:
                    (actor_model.h_n_dict[agent_id], actor_model.c_n_dict[agent_id]) = repackage_hidden((actor_model.h_n_dict[agent_id], actor_model.c_n_dict[agent_id]))
            actions = np.zeros(actor_model.num_agents)
            states_variable = np.zeros((8, 84, 84))
            states = [resize(state,(84, 84)) for state in states]
            for i, s in enumerate(states):
                states_variable[2*i:2*(i+1), :, :] = s.T

            # print(states_variable.shape)
            states_variable = torch.from_numpy(states_variable).float()
            state_value = critic_model(Variable(states_variable).unsqueeze(0))
            for agent_id in actor_model.live_agents:
                state = states[agent_id]
                state = state.T
                actions[agent_id] = select_action(state, agent_id, state_value)
            states, rewards, done, info_dict = env.step(actions)
            done_list = info_dict["done"]
            if args.render:
                env.render()
            for agent_id in actor_model.live_agents:
                if done_list[agent_id]:
                    actor_model.live_agents.remove(agent_id)
                actor_model.rewards_dict[agent_id].append(rewards[agent_id])
            if done:
                break
        total_reward = np.sum([np.sum(actor_model.rewards_dict[idx]) for idx in range(actor_model.num_agents)])

        running_reward = running_reward * 0.99 + total_reward * 0.01
        loss = finish_episode()
        queue.append(total_reward)
        print(total_reward, np.mean(queue))

        logger.scalar_summary('loss', loss.data[0], i_episode)
        logger.scalar_summary('reward', total_reward, i_episode)

        if total_reward > max_reward:
            max_reward = total_reward
            is_best = True
        else:
            is_best = False
        if i_episode % args.log_interval == 0:
            save_checkpoint({
                        'episode': i_episode,
                        'state_dict': actor_model.state_dict(),
                        'best_reward': max_reward,
                        'optimizer': actor_optimizer.state_dict(),
                    }, is_best, filename='policy_' + str(i_episode) + '.pth.tar')
            save_checkpoint({
                        'episode': i_episode,
                        'state_dict': critic_model.state_dict(),
                        'best_reward': max_reward,
                        'optimizer': critic_optimizer.state_dict(),
                    }, is_best, filename='critic_' + str(i_episode) + '.pth.tar')


        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast Reward: {:5f}\tBest Reward: {:.2f}'.format(
                i_episode, total_reward, max_reward))


if __name__ == '__main__':
    main()
