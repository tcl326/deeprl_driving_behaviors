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

from replay_buffer import ReplayBuffer
# import cv2


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tao', type=float, default=0.01,
                    help='target update percentage (default: 0.01)')
parser.add_argument('--buffer', type=float, default=1e3,
                    help='buffer size (default: 1000)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 100)')
args = parser.parse_args()


env = gym.make('Traffic-Multi-preset-cli-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorNetwork(nn.Module):
    def __init__(self, num_agents):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, (8,8), (4,4))
        self.conv2 = nn.Conv2d(32, 16, (4,4), (2,2))
        self.conv3 = nn.Conv2d(16, 16, (3,3), (1,1))
        self.rnn1 = nn.LSTM(input_size=784, hidden_size=256, num_layers=1)
        self.action1 = nn.Linear(256, 128)
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
        flatten = conv3.view(self.time_steps,1,-1)
        if self.reset or not self.h_n_dict[agent_id] or not self.c_n_dict[agent_id]:
            lstm1, (self.h_n_dict[agent_id], self.c_n_dict[agent_id]) = self.rnn1(flatten)
        else:
            lstm1, (self.h_n_dict[agent_id], self.c_n_dict[agent_id]) = self.rnn1(flatten, (self.h_n_dict[agent_id], self.c_n_dict[agent_id]))
            self.reset = False
        # print(flatten.size())
        flatten = lstm1.view(self.time_steps, -1)
        action = F.relu(self.action1(flatten))
        action = F.relu(self.action2(action))
        action_scores = self.action_head(action)
        return F.softmax(action_scores, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, num_agents):
        super(CriticNetwork, self).__init__()
        self.action1 = nn.Linear(3*num_agents, 64)
        self.action2 = nn.Linear(64, 128)
        self.conv1 = nn.Conv2d(2*num_agents, 64, (8,8), (4,4))
        self.conv2 = nn.Conv2d(64, 32, (4,4), (2,2))
        self.conv3 = nn.Conv2d(32, 16, (3,3), (1,1))
        # self.conv4 = nn.Conv2d(64, 512, (7,7), (1,1))
        self.rnn1 = nn.LSTM(input_size=784+128, hidden_size=256, num_layers=1)

        self.value1 = []
        self.value2 = []
        self.value_head = []
        for n in range(num_agents):
            self.value1.append(nn.Linear(256, 128))
            self.value2.append(nn.Linear(128, 64))
            self.value_head.append(nn.Linear(64, 1))

        self.h_n = None
        self.c_n = None
        self.reset = False
        self.time_steps = 1

    def forward(self, x, actions):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # print(conv3)
        flatten_conv = conv3.view(self.time_steps,1,-1)

        action1 = F.relu(self.action1(actions))
        action2 = F.relu(self.action2(action1))

        flatten_action = action2.view(self.time_steps, 1, -1)
        # print(flatten, actions.view(self.time_steps, 1, -1))
        concat = torch.cat((flatten_conv, flatten_action), 2)
        # print(concat.size())
        if self.reset or not self.h_n or not self.c_n:
            lstm1, (self.h_n, self.c_n) = self.rnn1(concat)
        else:
            lstm1, (self.h_n, self.c_n) = self.rnn1(concat, (self.h_n, self.c_n))
            self.reset = False
        # print(flatten.size())
        flatten = lstm1.view(self.time_steps, -1)
        value = []
        state_values = []
        for i in range(len(self.value1)):
            value.append(F.relu(self.value1[i](flatten)))
        for i in range(len(self.value2)):
            value[i] = F.relu(self.value2[i](value[i]))
        for i in range(len(self.value_head)):
            state_values.append(self.value_head[i](value[i]))
        return state_values

actor_model = ActorNetwork(num_agents=4)
print(actor_model)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.001)

critic_model = CriticNetwork(num_agents=4)
print(critic_model)
critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.001)

buffer = ReplayBuffer(args.buffer)



def get_prob(state, agent_id):
    # if agent_id == 0:
    # print(state.shape)
    # cv2.imshow('image' + str(agent_id),state[0,:,:] + state[1,:,:])
    # cv2.waitKey(1)
    state = torch.from_numpy(state).float()
    probs = actor_model(Variable(state).unsqueeze(0), agent_id)

    return probs
    # return action.data[0]


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
        # print(rewards_dict[agent_id])
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
    return actor_loss, critic_loss

def replay_memory(batch_size):
    states_whole_batch, actions_whole_batch, rewards_whole_batch, next_states_whole_batch, done_list_whole_batch = buffer.sample(batch_size)
    policy_losses = []
    value_losses = []
    for states_whole, actions_whole, rewards_whole, next_states_whole, done_list_whole in zip(states_whole_batch, actions_whole_batch, rewards_whole_batch, next_states_whole_batch, done_list_whole_batch):
        # print(rewards_whole.shape)
        probs = {agent_id:None for agent_id in range(actor_model.num_agents)}
        rewards_dict = {num:[] for num in range(actor_model.num_agents)}
        actor_model.time_steps = len(actions_whole)
        critic_model.time_steps = len(actions_whole)
        for agent_id in range(actor_model.num_agents):
            states = torch.from_numpy(states_whole[:, 2*agent_id:2*(agent_id+1),:,:]).float()
            probs[agent_id] = actor_model(Variable(states), agent_id)

        state_values = critic_model(Variable(torch.from_numpy(states_whole).float()), Variable(torch.from_numpy(np.array(actions_whole)).float()))[0]

        for agent_id in range(actor_model.num_agents):
            R = 0
            for r in rewards_whole[:,agent_id][::-1]:
                R = r + args.gamma * R
                rewards_dict[agent_id].insert(0, R)
            rewards_dict[agent_id] = torch.Tensor(rewards_dict[agent_id])
            rewards_dict[agent_id] = (rewards_dict[agent_id] - rewards_dict[agent_id].mean()) / (rewards_dict[agent_id].std() + np.finfo(np.float32).eps)
            # print(rewards_dict[agent_id])
            # pdb.set_trace()
            for log_prob, value, r in zip(probs[agent_id], state_values, rewards_dict[agent_id]):
                reward = r - value.data
                # print(value)
                # # reward = np.array([reward])
                # print(log_prob.size())
                # print(value.size())
                # # print(reward)
                # print(r)
                policy_losses.append(-log_prob * Variable(reward))
                value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    # print(value_losses)
    # print(policy_losses)
    actor_loss = torch.stack(policy_losses).sum()/float(actor_model.num_agents)/float(batch_size)
    actor_loss.backward()

    critic_loss = torch.stack(value_losses).sum()/float(actor_model.num_agents)/float(batch_size)
    critic_loss.backward()

    actor_optimizer.step()
    critic_optimizer.step()

    actor_model.rewards_dict.clear()
    actor_model.saved_actions_dict.clear()
    return actor_loss, critic_loss

def save_checkpoint(state, is_best, folder='model/multiple_central_critic_buffer_no_random', filename='checkpoint.pth.tar'):
    print("save checkpoint")

    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'model_best.pth.tar'))

def load_checkpoint(model, folder='model/multiple_central_critic_buffer', filename='checkpoint.pth.tar'):
    filename = os.path.join(folder, filename)
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

actor_model = load_checkpoint(actor_model, filename='policy_170.pth.tar')
critic_model = load_checkpoint(critic_model, filename='critic_170.pth.tar')

def main():
    queue = deque([], maxlen=10)
    log_path = 'log'
    from datetime import datetime
    now = datetime.now()
    log_path = "log/multiple_central_critic_buffer"
    logger = Logger(log_path,  now.strftime("%Y%m%d-%H%M%S"))
    is_best = False
    running_reward = 0
    max_reward = -float('inf')
    BPTT = 30
    batch_size = 10
    burn_in = 30
    for i_episode in count(1):
        states = env.reset()
        states = [resize(state,(84, 84)) for state in states]
        actor_model.live_agents = list(actor_model.agent_ids)
        actor_model.saved_actions_dict = {num:[] for num in range(actor_model.num_agents)}
        actor_model.rewards_dict = {num:[] for num in range(actor_model.num_agents)}
        actor_model.time_steps = 1
        actor_model.reset = True
        critic_model.reset = True
        states_whole = []
        actions_whole = []
        rewards_whole = []
        next_states_whole = []
        done_list_whole = []
        for t in range(10000):  # Don't infinite loop while learning
            if t % BPTT == 0 and t != 0:
                # (critic_model.h_n, critic_model.c_n) = repackage_hidden((critic_model.h_n, critic_model.c_n))
                for agent_id in actor_model.live_agents:
                    (actor_model.h_n_dict[agent_id], actor_model.c_n_dict[agent_id]) = repackage_hidden((actor_model.h_n_dict[agent_id], actor_model.c_n_dict[agent_id]))
            actions = np.zeros(actor_model.num_agents)
            probs = []
            actions_replay = [Variable(torch.from_numpy(np.array([0])).long()) for agent in range(actor_model.num_agents)]
            states_variable = np.zeros((8, 84, 84))

            for i, s in enumerate(states):

                states_variable[2*i:2*(i+1), :, :] = s.T

            # print(states_variable.shape)
            # states_variable = torch.from_numpy(states_variable).float()

            for agent_id in actor_model.live_agents:
                state = states[agent_id]

                state = state.T
                probs.append(get_prob(state, agent_id))

            for i, agent_id in enumerate(actor_model.live_agents):
                m = Categorical(probs[i])
                action = m.sample()
                # print(action)
                actions_replay[agent_id] = action
                actions[agent_id] = action.data[0]
            # state = torch.from_numpy(state).float()
            one_hot_actions = []
            for a in actions:
                c = [0,0,0]
                c[int(a)] = 1
                one_hot_actions += c
            # print(one_hot_actions)
            one_hot_actions = np.array(one_hot_actions)
            # state_values = critic_model(Variable(torch.from_numpy(states_variable).float()).unsqueeze(0), Variable(torch.from_numpy(one_hot_actions).float()).unsqueeze(0))

            # for agent_id in range(actor_model.num_agents):
            #     actor_model.saved_actions_dict[agent_id].append(SavedAction(m.log_prob(actions_replay[agent_id]), state_values[agent_id]))

            states_next, rewards, done, info_dict = env.step(actions)

            states_next = [resize(state,(84, 84)) for state in states_next]


            done_list = info_dict["done"]
            collision_list = info_dict["collision"]

            done = np.any(collision_list) or done

            if args.render:
                env.render()
            for agent_id in actor_model.live_agents:
                if done_list[agent_id]:
                    actor_model.live_agents.remove(agent_id)
                actor_model.rewards_dict[agent_id].append(rewards[agent_id])
            if done:
                break

            states_whole.append(states_variable)
            actions_whole.append(one_hot_actions)
            rewards_whole.append(rewards)
            next_states_whole.append(states_next)
            done_list_whole.append(done_list)

            states = states_next

        buffer.add(np.array(states_whole), np.array(actions_whole), np.array(rewards_whole), np.array(next_states_whole), np.array(done_list_whole))

        total_reward = np.sum([np.sum(actor_model.rewards_dict[idx]) for idx in range(actor_model.num_agents)])

        running_reward = running_reward * 0.99 + total_reward * 0.01

        # actor_loss, critic_loss = finish_episode()
        if i_episode > burn_in:
            actor_loss, critic_loss = replay_memory(batch_size)
            logger.scalar_summary('actor_loss', actor_loss.data[0], i_episode)
            logger.scalar_summary('critic_loss', critic_loss.data[0], i_episode)
            logger.scalar_summary('reward', total_reward, i_episode)

        queue.append(total_reward)
        print(total_reward, np.mean(queue))

        if np.mean(queue) > max_reward:
            max_reward = np.mean(queue)
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
