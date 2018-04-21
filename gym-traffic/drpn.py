import argparse
import gym
import gym_traffic
import numpy as np
from itertools import count
from collections import namedtuple
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


env = gym.make('Traffic-Multi-gui-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self, num_agents):
        super(Policy, self).__init__()
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
        self.value1 = nn.Linear(512, 128)
        self.value2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
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
        value = F.relu(self.value1(flatten))
        value = F.relu(self.value2(value))
        state_values = self.value_head(value)
        # print(state_values)
        return F.softmax(action_scores, dim=-1), state_values


model = Policy(num_agents=4)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def select_action(state, agent_id):
    state = torch.from_numpy(state).float()
    probs, state_value = model(Variable(state).unsqueeze(0), agent_id)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions_dict[agent_id].append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]


def finish_episode():
    saved_actions_dict = model.saved_actions_dict
    rewards_dict = {num:[] for num in range(model.num_agents)}
    policy_losses = []
    value_losses = []

    for agent_id in range(model.num_agents):
        R = 0
        for r in model.rewards_dict[agent_id][::-1]:
            R = r + args.gamma * R
            rewards_dict[agent_id].insert(0, R)
        rewards_dict[agent_id] = torch.Tensor(rewards_dict[agent_id])
        rewards_dict[agent_id] = (rewards_dict[agent_id] - rewards_dict[agent_id].mean()) / (rewards_dict[agent_id].std() + np.finfo(np.float32).eps)
        # pdb.set_trace()
        for (log_prob, value), r in zip(saved_actions_dict[agent_id], rewards_dict[agent_id]):
            reward = r - value.data[0]
            policy_losses.append(-log_prob * Variable(reward))
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    # print(value_losses)
    # print(policy_losses)
    loss = torch.stack(policy_losses).sum()/float(model.num_agents) + torch.stack(value_losses).sum()/float(model.num_agents)
    loss.backward()
    optimizer.step()
    model.rewards_dict.clear()
    model.saved_actions_dict.clear()
    return loss

def save_checkpoint(state, is_best, filename='model/multiple_recurrent/checkpoint.pth.tar'):
    print("save checkpoint")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model/multiple_recurrent/model_best.pth.tar')

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def main():
    log_path = 'log'
    logger = Logger(log_path, 'multiple_recurrent')
    is_best = False
    running_reward = 0
    max_reward = -float('inf')
    BPTT = 10
    for i_episode in count(1):
        states = env.reset()
        model.live_agents = list(model.agent_ids)
        model.saved_actions_dict = {num:[] for num in range(model.num_agents)}
        model.rewards_dict = {num:[] for num in range(model.num_agents)}
        model.reset = True
        for t in range(10000):  # Don't infinite loop while learning
            if t % BPTT == 0 and t != 0:
                for agent_id in model.live_agents:
                    (model.h_n_dict[agent_id], model.c_n_dict[agent_id]) = repackage_hidden((model.h_n_dict[agent_id], model.c_n_dict[agent_id]))
            actions = np.zeros(model.num_agents)
            for agent_id in model.live_agents:
                state = states[agent_id]
                state = resize(state,(84, 84))
                state = state.T
                actions[agent_id] = select_action(state, agent_id)
            states, rewards, done, info_dict = env.step(actions)
            done_list = info_dict["done"]
            if args.render:
                env.render()
            for agent_id in model.live_agents:
                if done_list[agent_id]:
                    model.live_agents.remove(agent_id)
                model.rewards_dict[agent_id].append(rewards[agent_id])
            if done:
                break
        total_reward = sum([np.mean(model.rewards_dict[idx]) for idx in range(model.num_agents)])
        print(total_reward)
        running_reward = running_reward * 0.99 + total_reward * 0.01
        loss = finish_episode()



        logger.scalar_summary('loss', loss.data[0], i_episode)
        logger.scalar_summary('reward', total_reward, i_episode)

        if total_reward > max_reward:
            max_reward = total_reward
            is_best = True
        else:
            is_best = False

        save_checkpoint({
                    'episode': i_episode,
                    'state_dict': model.state_dict(),
                    'best_reward': max_reward,
                    'optimizer': optimizer.state_dict(),
                }, is_best)

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast Reward: {:5f}\Best Reward: {:.2f}'.format(
                i_episode, total_reward, max_reward))


if __name__ == '__main__':
    main()
