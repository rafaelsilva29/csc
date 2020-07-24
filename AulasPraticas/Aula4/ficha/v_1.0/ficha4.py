import gym
import random
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

# Need to have because of macos system
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

logging.getLogger('tensorflow').setLevel(logging.ERROR)
#for replicability purposes set seed
#TODO: set random seed
tf.random.set_seed(91195003)
#for an easy reset backend session state
tf.keras.backend.clear_session()

#=====================================================================================
#
#                            Player Agent
#
#=====================================================================================
class PlayerAgent:

  '''
  Constructor - defining game variables
  '''
  def __init__(self, game_steps, nr_of_games_for_training, score_requirement, game='CartPole-v1', log=False):
    self.game_steps = game_steps
    self.nr_of_games_for_training = nr_of_games_for_training
    self.score_requirement = score_requirement
    self.random_games = 50 
    self.log = log 
    self.env = gym.make(game)
    self.scores = []
  
  '''
  Let us test the environment and play some random games!
  '''
  def play_random_games(self):
    for game in range(self.random_games):
      self.env.reset()
      for step in range(self.game_steps):
        if self.log:
          self.env.render()
        action = self.env.action_space.sample()
        observation, reward, done, info = self.env.step(action)
        if done:
          print("Game finisihed after {} steps".format(step+1))
          break
  
  '''
  To build our training set we must play some games!
  Let us start by playing completly random games and save those where we achieve an interesting score.
  You should then use this training set to train the model!!!
  '''
  def build_training_set(self):
    training_set = []
    score_set = []

    for game in range(self.nr_of_games_for_training):
      cumulative_game_score = 0
      game_memory = []
      obs_prev = self.env.reset()
      

      for step in range(self.game_steps):
        #env.render()
        action = self.env.action_space.sample()
        obs_next, reward, done, info = self.env.step(action)
        game_memory.append([action, obs_prev])
        cumulative_game_score += reward
        obs_prev = obs_next
        if done: 
          #print("Game finished after {} steps".format(steps+1))
          break
      
      #the game is finished. Was it a decent game?
      if cumulative_game_score > self.score_requirement:
        score_set.append(cumulative_game_score)
        for play in game_memory:
          if play[0] == 0:
            one_hot_action = [1, 0]
          elif play[0] == 1:
            one_hot_action = [0, 1]
          
          training_set.append([one_hot_action, play[1]])
          
    #just in case you wanted to reference later
    training_set_save = np.array(training_set)
    np.save('training_set_saved.npy',training_set_save)
    
    #print some stats
    if score_set:
      print('Average score:', np.mean(score_set))
      print('Number of stored games per score', Counter(score_set))
    
    return training_set
  
  '''
  Let us play the game after been trained to do that!
  '''
  def play_the_game(self, flag, train_flag):
    #build training set
    if train_flag:
        training_set = self.build_training_set()
    else:
        training_set = np.load('training_set_saved.npy')
        
    print(training_set)

    #using the model
    if flag:
        mlp = MLP()

    mlp.build()
    mlp.fit(training_set)
    
    choices = []
    game_memory = []

    for game in range(self.nr_of_games_for_training):
      score = 0
      obs_prev = self.env.reset()
      done = False

      while not done:
        if self.log:
          self.env.render()
        action = mlp.predict(obs_prev)
        choices.append(action)
        obs_next, reward, done, _ = self.env.step(action)
        score += reward
        move = []
        if action == 0:
            move = [1, 0]
        else:
            move = [0, 1]
        game_memory.append([obs_next, move])
        obs_prev = obs_next
    
      game_memory_save = np.array(game_memory)
      np.save('game_memory.npy',game_memory_save)
      
      self.scores.append(score)
      print('Current score: ' + str(score) + '; Average score:', sum(self.scores)/len(self.scores))


#=====================================================================================
#
#                            Multi-Layer Perceptron
#
#=====================================================================================
class MLP:

  '''
  Constructor - defining game variables
  '''
  def __init__(self, epochs=10, batch_size=32, output_neurons=2):
    self.epochs = epochs
    self.batch_size = batch_size
    self.output_neurons = output_neurons
  
  def prepare_data(self, training_set):
    #reshape X to list of inputs
    X = np.array([i[1] for i in training_set], dtype="float32").reshape(-1, len(training_set[0][1]))
    #y are the actions
    y = np.array([i[0] for i in training_set])
    return X, y


  def build(self):
    model = Sequential()
    model.add(Dense(128, input_shape=(4,), activation="relu"))
    model.add(Dropout(0.8))
    
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.8))
    
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.8))
    
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.8))
    
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.8))
    model.add(Dense(2, activation="softmax"))
    
    model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["categorical_accuracy"])
    
    return model


  def fit(self, training_set):
    X, y = self.prepare_data(training_set)
    
    model = self.build()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='ckpt/my_model_{epoch}_{loss:.3f}.hdf5', 
            monitor='loss', 
            verbose=1, 
            save_best_only=True,
            save_weights_only=False, 
        ) 
    ]
    
    model.fit(X, y, epochs=10, callbacks=callbacks)

  
  def predict(self, obs):
    for each_game in range(75):
        prev_obs = []
        for step_index in range(500):
            #env.render()
            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(obs.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            
            return action
