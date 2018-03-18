# from keras.models import Model
# from keras.layers import Dense, Flatten, LeakyReLU, Input, merge, Reshape, Lambda, BatchNormalization, Dropout
# from keras.regularizers import L1L2Regularizer
# from keras.utils.np_utils import to_categorical
import numpy as np
from gym import spaces
# from keras.optimizers import Adam
import itertools
# from keras import backend as K
import os
from .agent import Agent
from random import shuffle
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc


# def flatten_spaces(space):
#     if isinstance(space, spaces.Tuple):
#         return list(itertools.chain.from_iterable(flatten_spaces(s) for s in space.spaces))
#     else:
#         return [space]


# def calc_input_dim(space):
#     dims = []
#     print "Space: {}".format(space)
#     print "Flattened: {}".format(flatten_spaces(space))
#     for i in flatten_spaces(space):
#         if isinstance(i, spaces.Discrete):
#             dims.append(i.n)
#         elif isinstance(i, spaces.Box):
#             dims.append(np.prod(i.shape))
#         else:
#             raise NotImplementedError("Only Discrete and Box input spaces currently supported")
#     return np.sum(dims)


# def concat_input(observation, input_space):
#     if isinstance(input_space, spaces.Tuple):
#         return np.hstack([np.array(concat_input(obs, space)) for obs, space in
#                           zip(observation, input_space.spaces)])
#     elif isinstance(input_space, spaces.Discrete):
#         return to_categorical(observation, nb_classes=input_space.n).reshape((1, -1))
#     elif isinstance(input_space, spaces.Box):
#         return observation.reshape((1, -1))
#     else:
#         raise NotImplementedError("Only Discrete and Box input spaces currently supported")


class DRQN():
    def __init__(self, h_size, batch_size, rnn_cell, myScope):
        # super(DQN, self).__init__(input_space, action_space, seed=seed)
        # self.input_dim = calc_input_dim(input_space)
        # self.memory_size = memory_size
        # self.replay_size = replay_size
        # self.discount = K.variable(K.cast_to_floatx(discount))
        self.h_size = h_size
        self.rnn_cell = rnn_cell
        # self.step = 0
        self.height = 84
        self.width = 84
        # self.bs = batch_size
        # self.data_dim = self.input_dim * self.memory_size
        # self.replay = []
        # self.new_episode()
        # if optimizer is None:
        #     optimizer = Adam(1e-4, decay=1e-6)
        # self.optimizer = optimizer
        # if not isinstance(action_space, spaces.Discrete):
        #     raise NotImplementedError("Only Discrete action spaces supported")
        self.myScope = myScope
        self.build_network()


    # def new_episode(self):
    #     self.memory = [np.zeros((1, self.input_dim)) for i in range(self.memory_size)]
    #     self.observation = None
    #     self.last_observation = None

    def build_network(self):
        # self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)

        # self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])

        self.imageIn = tf.placeholder(shape=[None, self.height, self.width, 2], dtype = tf.float32)

        self.conv1 = slim.convolution2d(inputs = self.imageIn, num_outputs = 32, kernel_size = [8,8], stride = [4,4],
            padding = 'VALID', biases_initializer = None, scope = self.myScope + '_conv1')

        self.conv2 = slim.convolution2d(inputs = self.conv1, num_outputs = 64, kernel_size = [4,4], stride = [2,2],
            padding = 'VALID', biases_initializer = None, scope = self.myScope + '_conv2')

        self.conv3 = slim.convolution2d(inputs = self.conv2, num_outputs = 64, kernel_size = [3,3], stride = [1,1],
            padding = 'VALID', biases_initializer = None, scope = self.myScope + '_conv3')

        self.conv4 = slim.convolution2d(inputs = self.conv3, num_outputs = self.h_size, kernel_size = [7,7], stride = [1,1],
            padding = 'VALID', biases_initializer = None, scope = self.myScope + '_conv4')

        self.trainLength = tf.placeholder(dtype = tf.int32)

        #We take the output from the final convolutional layer and send it to a recurrent layer.
        #The input must be reshaped into [batch x trace x units] for rnn processing,
        #and then returned to [batch x units] when sent through the upper levles.

        self.batch_size = tf.placeholder(dtype = tf.int32, shape = [])


        self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.trainLength, self.h_size])

        self.state_in = self.rnn_cell.zero_state(self.batch_size, tf.float32)

        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs = self.convFlat, cell = self.rnn_cell, dtype = tf.float32,
            initial_state = self.state_in, scope = self.myScope + '_rnn')

        self.rnn = tf.reshape(self.rnn,shape=[-1, self.h_size])

        #The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        # self.AW = tf.Variable(tf.random_normal([self.h_size//2, 4]))
        self.AW = tf.Variable(tf.random_normal([self.h_size//2, 3]))
        self.VW = tf.Variable(tf.random_normal([self.h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.salience = tf.gradients(self.Advantage, self.imageIn)
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis = 1, keep_dims = True))
        self.predict = tf.argmax(self.Qout, 1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 3, dtype = tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis = 1)

        self.td_error = tf.square(self.targetQ - self.Q)

        #In order to only propogate accurate gradients through the network, we will mask the first
        #half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size, self.trainLength//2])
        self.maskB = tf.ones([self.batch_size, self.trainLength//2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)




    # def observe(self, observation):
    #     observation = concat_input(observation, self.input_space)
    #     self.memory = self.memory[1:] + [observation]
    #     self.last_observation = self.observation
    #     self.observation = np.hstack(self.memory)

    # def act(self): ## take the observation and run through the graph to predict the action
    #     action = self.predict
    #     return action

    # def combined_replay(self):
    #     return [np.vstack(x[i] for x in self.replay) for i in range(5)]

    # def learn(self, action, reward, done):
    #     datum = [self.last_observation, action, self.observation, [[1]] if done else [[0]], reward]
    #     self.replay.append(datum)
    #     if len(self.replay) > self.replay_size:
    #         self.replay.pop(0)
    #     # shuffle(self.replay)
    #     data = self.combined_replay()
    #     loss = self.training_model.train_on_batch(data[0:4], data[4])

    #     return loss

    # def save(self, filepath):
    #     dirpath = os.path.dirname(filepath)
    #     if not os.path.exists(dirpath):
    #         os.makedirs(dirpath)
    #     self.Q.save_weights(filepath)

    # def load(self, filepath):
    #     self.Q.load_weights(filepath)
