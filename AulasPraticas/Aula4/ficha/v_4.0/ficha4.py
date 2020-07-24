import gym
import random
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
     
        # just in case you wanted to reference later
        training_set_save = np.array(training_set)
        np.save('training_set.npy',training_set_save)
    
    #print some stats
    if score_set:
      print('Average score:', np.mean(score_set))
      print('Number of stored games per score', Counter(score_set))
    
    return training_set
  
  '''
  Let us play the game after been trained to do that!
  '''
  def play_the_game(self):
    #build training set
    training_set = self.build_training_set()
    
    #load a training_set
    #training_set = np.load('training_set.npy')
    
    #load a player_set
    #training_set = np.load('player_set.npy')

    #using the model
    mlp = MLP()
    mlp.fit(training_set)
    
    self.env._max_episode_steps = self.game_steps
    
    game_memory = []
    player_set = []

    for game in range(self.nr_of_games_for_training):
      score = 0
      obs_prev = []
      self.env.reset()
      done = False

      while not done:
        if self.log:
          self.env.render()
        action = mlp.predict(obs_prev)
        obs_next, reward, done, _ = self.env.step(action)
        game_memory.append([action, obs_prev])
        score += reward
        obs_prev = obs_next
     
      self.scores.append(score)
      
      #the game is finished. Was it a decent game?
      if (max(self.scores) > self.score_requirement) and (score >= max(self.scores)):
          for play in game_memory:
              if play[0] == 0:
                  one_hot_action = [1, 0]
              elif play[0] == 1:
                  one_hot_action = [0, 1]
  
          player_set.append([one_hot_action, play[1]])
          
          #Save good game
          print("Save game...")
          player_set_save = np.array(player_set)
          np.save('player_set.npy',player_set_save)

      print('Current score: ' + str(score) + '; Average score: ' + str(sum(self.scores)/len(self.scores)) + '; Max score:', max(self.scores))



#=====================================================================================
#
#                            Multi-Layer Perceptron
#
#=====================================================================================
class MLP:

  '''
  Constructor - defining game variables
  '''
  def __init__(self, epochs=5, batch_size=128, output_neurons=10):
    self.epochs = epochs
    self.batch_size = batch_size
    self.output_neurons = output_neurons
    self.model = ""
  
  def prepare_data(self, training_set):
    #reshape X to list of inputs
    X = np.array([i[1] for i in training_set], dtype="float32").reshape(-1, len(training_set[0][1]))
    #y are the actions
    y = np.array([i[0] for i in training_set])
    return X, y


  def build(self, input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    
    model.compile(
        learning_rate=1e-3,
        loss='mse',
        optimizer="adam",
        metrics=["accuracy"])
    
    return model

  def fit(self, training_set):
    X, y = self.prepare_data(training_set)

    self.model = self.build(input_size=len(X[0]), output_size=len(y[0]))
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='ckpt/my_model_{epoch}_{loss:.3f}.hdf5', 
            monitor='loss', 
            verbose=1, 
            save_best_only=True,
            save_weights_only=False, 
        ) 
    ]
    
    history = self.model.fit(X, y, epochs=10, callbacks=callbacks)
    print("History:",history.history.keys())
    
  
  def predict(self, obs):
    for step_index in range(500):
        if len(obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(self.model.predict(obs.reshape(-1,len(obs))))
        
        return action




#=====================================================================================
#
#                            Main Execution
#
#=====================================================================================

playerAgent = PlayerAgent(500, 1000, 75, game='CartPole-v1', log=True)
#playerAgent.play_random_games()
playerAgent.play_the_game()
print("...Finnish...")