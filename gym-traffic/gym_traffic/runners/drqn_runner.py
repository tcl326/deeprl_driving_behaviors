import numpy as np
from tqdm import tqdm
import tensorflow as tf
import imageio
imageio.plugins.ffmpeg.download()
from gym_traffic.utils.helper import *
from gym_traffic.agents.drqn import DRQN
from IPython import embed
from skimage.transform import resize
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
    self.batch_size = 16 #How many experience traces to use for each training step.
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

      updateTarget(targetOps,sess)
      for i in range(self.num_episodes):
        episodeBuffer = []
        print ('Episode: ', i)
        sP = env.reset()
        s = [None] * 4
        for v in range(4):
          s[v] = resize(sP[v], (84, 84))
        #s = sP
        d = [False] * 4
        rAll = 0
        j = 0
        state = [[np.zeros([1, self.h_size]),np.zeros([1, self.h_size])]] * 4
        state1 = [None] * 4
        a = [None]*4
        while j < self.max_epLength:
          j+=1
          for v in range(4):
	          if np.random.rand(1) < e or total_steps < self.pre_train_steps:
	            state1[v] = sess.run(mainQN.rnn_state,
	              feed_dict={mainQN.imageIn:[s[v]/255.0],mainQN.trainLength:1,mainQN.state_in:state[v],mainQN.batch_size:1})
	            
	            a[v] = np.random.randint(0,3)
	            assert(a[v]<3)
	          else:
	            a[v], state1[v] = sess.run([mainQN.predict,mainQN.rnn_state],
	              feed_dict={mainQN.imageIn:[s[v]/255.0],mainQN.trainLength:1,mainQN.state_in:state[v],mainQN.batch_size:1})
	            a[v] = a[v][0]
	            assert(a[v]<3)
          #print(a)
          d_old = d.copy()
          s1P, r, d, info = env.step(a)
          #print(d)
          s1 = [None] * 4
          for v in range(4):
            s1[v] = resize(s1P[v], (84, 84))
          #s1 = s1P
          total_steps += 1
          for v in range(4):
            if not d_old[v]:
              episodeBuffer.append(np.reshape(np.array([s[v],a[v],r[v],s1[v],d[v]]),[1,5]))
           # print(episodeBuffer[-1].shape)
          if total_steps > self.pre_train_steps:
            if e > self.endE:
              e -= stepDrop

            if total_steps % (self.update_freq) == 0:
              updateTarget(targetOps,sess)
              state_train = (np.zeros([self.batch_size, self.h_size]),np.zeros([self.batch_size, self.h_size]))

              trainBatch = myBuffer.sample(self.batch_size, self.trace_length)
              trainBatch_st_0 = np.concatenate([arr[np.newaxis] for arr in trainBatch[:,0]])
              trainBatch_st_1 = np.concatenate([arr[np.newaxis] for arr in trainBatch[:,3]])

              Q1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:trainBatch_st_1/255.0,
                mainQN.trainLength: self.trace_length, mainQN.state_in: state_train, mainQN.batch_size: self.batch_size})

              Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.imageIn:trainBatch_st_1/255.0,
                targetQN.trainLength: self.trace_length, targetQN.state_in:state_train, targetQN.batch_size: self.batch_size})

              end_multiplier = -(trainBatch[:,4] - 1)
              doubleQ = Q2[range(self.batch_size * self.trace_length), Q1]
              targetQ = trainBatch[:,2] + (self.y*doubleQ * end_multiplier)
              
              sess.run(mainQN.updateModel, feed_dict={mainQN.imageIn: trainBatch_st_0/255.0,
                mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:,1], mainQN.trainLength: self.trace_length,
                mainQN.state_in: state_train, mainQN.batch_size: self.batch_size})

       	  
          rAll += np.array([r[i] * (not d_old[i]) for i in range(4)])
          s = s1.copy()
          sP = s1P.copy()
          state = state1.copy()
          if (d == True).all():
            break

        print ('steps taken: ', j)
        print ('total reward: ', np.sum(rAll))
	       
             
        if (len(episodeBuffer)>= self.trace_length):
          bufferArray = np.array(episodeBuffer)
          episodeBuffer = list(zip(bufferArray))
          myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll[0])

        #Periodically save the model.
        if i % 100 == 0 and i != 0:
            saver.save(sess,self.path+'/model-'+str(i)+'.cptk')
            print ("Saved Model")
        if len(rList) % self.summaryLength == 0 and len(rList) != 0:
            print (total_steps,np.mean(rList[-self.summaryLength:]), e)
            saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer), [len(episodeBuffer),5]), self.summaryLength,
              self.h_size, sess, mainQN, self.time_per_step)
      saver.save(sess,self.path+'/model-'+str(i)+'.cptk')
