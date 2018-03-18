import numpy as np
from tqdm import tqdm
import tensorflow as tf
import imageio
imageio.plugins.ffmpeg.download()
from gym_traffic.utils.helper import *
from gym_traffic.agents.drqn import DRQN
from IPython import embed

class experience_buffer():
  def __init__(self, buffer_size = 500):
    self.buffer = []
    self.buffer_size = buffer_size

  def add(self,experience):
    if len(self.buffer) + 1 >= self.buffer_size:
      self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
    self.buffer.append(experience)
    print ('buffer_size: ', len(self.buffer))

  def sample(self,batch_size,trace_length):
    # print ('np random sample: ', 'self.buffer: ', len(self.buffer), 'batch_size: ', batch_size)
    sampled_episodes = random.sample(self.buffer,batch_size)
    sampledTraces = []
    for episode in sampled_episodes:
      point = np.random.randint(0,len(episode)+1-trace_length)
      sampledTraces.append(episode[point:point+trace_length])
    sampledTraces = np.array(sampledTraces)
    # print ('!!!!!!!!!', sampledTraces.shape)
    return np.reshape(sampledTraces,[batch_size*trace_length,5])


class DRQNRunner(object):

  def __init__(self, max_steps_per_episode = 1000):
    # self.max_steps_per_episode=max_steps_per_episode
          #Setting the training parameters
    self.batch_size = 4 #How many experience traces to use for each training step.
    self.trace_length = 8 #How long each experience trace will be when training
    self.update_freq = 5 #How often to perform a training step.
    self.y = .99 #Discount factor on the target Q-values
    self.startE = 1 #Starting chance of random action
    self.endE = 0.1 #Final chance of random action
    self.anneling_steps = 10000 #How many steps of training to reduce startE to endE.
    self.num_episodes = 10000 #How many episodes of game environment to train network with.
    self.pre_train_steps = 10000 #How many steps of random actions before training begins.
    self.load_model = False #Whether to load a saved model.
    self.path = "./drqn" #The path to save our model to.
    self.h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    self.max_epLength = max_steps_per_episode #The max allowed length of our episode.
    self.time_per_step = 1 #Length of each step used in gif creation
    self.summaryLength = 100 #Number of epidoes to periodically save for analysis
    self.tau = 0.001


  def run_training(self, env):
    tf.reset_default_graph()

    #We define the cells for the primary and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.h_size,state_is_tuple = True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units = self.h_size,state_is_tuple = True)
    mainQN = DRQN(self.h_size, self.batch_size, cell, 'main')
    targetQN = DRQN(self.h_size, self.batch_size, cellT, 'target')

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=10)

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, self.tau)

    myBuffer = experience_buffer()

    #Set the rate of random action decrease.
    e = self.startE
    stepDrop = (self.startE - self.endE)/self.anneling_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    #Make a path for our model to be saved in.
    if not os.path.exists(self.path):
      os.makedirs(self.path)

    ##Write the first line of the master log-file for the Control Center
    with open('../Center/log.csv', 'w') as myfile:
      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
      wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])


    with tf.Session() as sess:
      if (self.load_model == True):
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(self.path)
        saver.restore(sess,ckpt.model_checkpoint_path)
      sess.run(init)

      updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
      for i in range(self.num_episodes):
        episodeBuffer = []
        # print ('-------------------', len(episodeBuffer), i, self.num_episodes)
        #Reset environment and get first new observation
        print ('Episode: ', i)
        # print ('len(myBuffer.buffer): ', len(myBuffer.buffer))
        sP = env.reset()
        # s = processState(sP)
        s = sP
        d = False
        rAll = 0
        j = 0
        state = (np.zeros([1, self.h_size]),np.zeros([1, self.h_size])) #Reset the recurrent layer's hidden state
        #The Q-Network
        while j < self.max_epLength:
          j+=1
          # print ('Episode: ', i, ' Step: ', j)
          #Choose an action greedily (with e chance of random action) from the Q-network
          if np.random.rand(1) < e or total_steps < self.pre_train_steps:
            state1 = sess.run(mainQN.rnn_state,
              feed_dict={mainQN.imageIn:[s/255.0],mainQN.trainLength:1,mainQN.state_in:state,mainQN.batch_size:1})
            #???????????????????????????????? normalized images ??????????????????????????????????????????#
            # a = np.random.randint(0,4)
            a = np.random.randint(0,3)
            assert(a<3)
          else:
            a, state1 = sess.run([mainQN.predict,mainQN.rnn_state],
              feed_dict={mainQN.imageIn:[s/255.0],mainQN.trainLength:1,mainQN.state_in:state,mainQN.batch_size:1})
            a = a[0]
            assert(a<3)
            #???????????????????????????????? normalized images ??????????????????????????????????????????#

          ## observation, reward, done, info = env.step(action)
          ### Does step return 4 things?

          s1P, r, d, info = env.step(a)
          # s1 = processState(s1P)
          s1 = s1P
          total_steps += 1
          episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
          if total_steps > self.pre_train_steps:
            if e > self.endE:
              e -= stepDrop

            if total_steps % (self.update_freq) == 0:
              # print ("------------- in --------------")
              updateTarget(targetOps,sess)
              #Reset the recurrent layer's hidden state
              state_train = (np.zeros([self.batch_size, self.h_size]),np.zeros([self.batch_size, self.h_size]))

              trainBatch = myBuffer.sample(self.batch_size, self.trace_length) #Get a random batch of experiences. (32,5)
              # trainBatch_stacked = tf.stack(trainBatch[:,3])
              # embed()
              # print ('trainBatch.shape: ', trainBatch.shape, 'trainBatch[:,3].shape: ', (trainBatch[:,3]/255.0).shape,
              #   'np.vstack(trainBatch[:,3]/255.0).shape', np.vstack(trainBatch[:,3]/255.0).shape, 'trainBatch_stacked.shape: ', trainBatch_stacked.shape)
              #Below we perform the Double-DQN update to the target Q-values
              trainBatch_st_0 = np.concatenate([arr[np.newaxis] for arr in trainBatch[:,0]])
              trainBatch_st_1 = np.concatenate([arr[np.newaxis] for arr in trainBatch[:,3]])

              Q1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:trainBatch_st_1/255.0,
                mainQN.trainLength: self.trace_length, mainQN.state_in: state_train, mainQN.batch_size: self.batch_size})

              Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.imageIn:trainBatch_st_1/255.0,
                targetQN.trainLength: self.trace_length, targetQN.state_in:state_train, targetQN.batch_size: self.batch_size})

              end_multiplier = -(trainBatch[:,4] - 1)
              doubleQ = Q2[range(self.batch_size * self.trace_length), Q1]
              targetQ = trainBatch[:,2] + (self.y*doubleQ * end_multiplier)
              #Update the network with our target values.
              sess.run(mainQN.updateModel, feed_dict={mainQN.imageIn: trainBatch_st_0/255.0,
                mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:,1], mainQN.trainLength: self.trace_length,
                mainQN.state_in: state_train, mainQN.batch_size: self.batch_size})

          rAll += r
          s = s1
          sP = s1P
          state = state1
          if d == True:
            # print ('********* DONE! *********')
            break

        print ('steps taken: ', j)
        print ('total reward: ', rAll)
        #Add the episode to the experience buffer
        if (len(episodeBuffer)>= self.trace_length):
          bufferArray = np.array(episodeBuffer)
          episodeBuffer = list(zip(bufferArray))
          myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        #Periodically save the model.
        if i % 100 == 0 and i != 0:
            saver.save(sess,self.path+'/model-'+str(i)+'.cptk')
            print ("Saved Model")
        if len(rList) % self.summaryLength == 0 and len(rList) != 0:
            print (total_steps,np.mean(rList[-self.summaryLength:]), e)
            saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer), [len(episodeBuffer),5]), self.summaryLength,
              self.h_size, sess, mainQN, self.time_per_step)
      saver.save(sess,self.path+'/model-'+str(i)+'.cptk')
