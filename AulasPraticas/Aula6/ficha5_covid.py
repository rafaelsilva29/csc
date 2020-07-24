#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:00:55 2020

@author: rafaelsilva
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.model_selection import TimeSeriesSplit
from numpy import nan
from numpy import isnan
from numpy import array


tf.random.set_seed(91195003)
np.random.seed(91195003)
#for an easy reset backend session state
tf.keras.backend.clear_session()


'''
Read csv file
'''
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8', index_col='row ID')
    df.index.name = 'Index'
    return df


'''
Fill missing values with a value at the same time one day ago
'''
def fill_missing(df):
    df['delay_in_seconds'].fillna(0, inplace=True)
    df['length_in_meters'].fillna(0, inplace=True)
    values = df.values
    time = 24 * 7
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if col == 9 or col == 12:
              time = 1
            else:
              time = 24 * 7
            if isnan(values[row, col]):
                values[row, col] = values[row - time, col]
    new_df = pd.DataFrame(values, columns=df.columns, index=df.index)
    return new_df


'''
Prepare dataset
'''
def prepare_data(df, norm_range=(-1,1)):
    # Drop columns
    columns = ['label_day','magnitude_criteria','atmosphere','description_criteria','rain','clouds','current_luminosity','cloudiness']
    data = df.drop(columns=columns)
    
    # Data until now
    index = str(now.hour) + '/' +  str(now.day) + '/' + str(now.month)
    df_until_now = data.loc[:index]

    # Make dataset numeric
    df_until_now = df_until_now.astype('float32')
    
    # Missing Value
    df_temp = fill_missing(df_until_now)
    #df_temp = df_temp.interpolate(method='linear', limit_direction='forward', axis=0)
    #print(df_temp.isnull().sum())

    # Normalized data
    scaler = MinMaxScaler(feature_range=norm_range)
    data_scaled = scaler.fit_transform(df_temp.values)
    new_df = pd.DataFrame(data_scaled,columns=df_temp.columns,index=df_temp.index)
    
    return new_df, scaler, df_temp, columns


'''
Creating a supervised problem
'''
def to_supervised(df, timesteps, multisteps, nr_features):
    data = df.values
    X, y = list(),list()
    data_size = len(data)
    for curr_pos in range(data_size):
        end_timesteps = curr_pos + timesteps
        begin_prev = end_timesteps
        end_prev = end_timesteps + multisteps
        if end_prev < data_size:
            X.append(data[curr_pos:end_timesteps,:])
            y.append(data[begin_prev:end_prev,5])
            
    X = np.reshape(np.array(X), (len(X), timesteps, nr_features))
    y = np.reshape(np.array(y), (len(y), multisteps))
        
    return X, y


'''
Buid the model 
'''
def build_model(timesteps, features, multisteps, n_neurons=32, activation='tanh', dropout_rate=0.4):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(int(n_neurons*2), input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(int(n_neurons), activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation=activation))
    model.compile(
            loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['mae']
        )
    
    return model


'''
Forecast
'''
def forecast(best_model, X, timesteps, multisteps, scaler, columns, nr_features, df_all):
     input_seq = X[-1:]
     inp = input_seq
     predictions = list()
     for step in range(1,multisteps+1):
        yhat = best_model.predict(inp, verbose=verbose)[0]
        yhat_inversed, inp = predict_multistep(yhat, scaler, columns, inp, timesteps, step, df_all)
        predictions.append(yhat_inversed)
     return predictions


'''
Predict multistep
'''
def predict_multistep(yhat, scaler, columns, inp, timesteps, step, df_all):
    # Save prediction(speed_diff) on data frame
    df_aux = pd.DataFrame(np.nan, index=range(0,1), columns=columns)
    df_aux['speed_diff'] = yhat[0]
    
    # Inverse scale of prediction
    yhat_inversed = scaler.inverse_transform(df_aux.values)[0][5]
    yhat_inversed = round(yhat_inversed, 2)
    
    # Concatenate data from lasts timestep and prediction, save on data frame
    inp = np.concatenate((inp, np.array(df_aux.values)[:,None]), axis=1)
    df_temp = pd.DataFrame(inp[0], index=range(0,timesteps+1), columns=columns)

    # Fill last multi_step with values df_all and prediction speed_diff    
    next_multi_step = datetime.datetime.today() + datetime.timedelta(hours=step)
    index = str(next_multi_step.hour) + '/' +  str(next_multi_step.day) + '/' + str(next_multi_step.month)
    df_temp.iloc[-1:] = scaler.transform(df_all.loc[[index]])
    df_temp.iloc[-1, df_temp.columns.get_loc('speed_diff')] = yhat[0]

    # Drop last timestep
    inp_temp = list()
    inp_temp.append(df_temp.values)
    inp = np.reshape(np.array(inp_temp), (len(inp_temp), timesteps+1, nr_features))
    inp = np.delete(inp, (0), axis=1)

    return yhat_inversed, inp


'''
Plot Loss
'''
def plot_loss(history, count):
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()


'''
Plot Mae
'''
def plot_mae(history, count):
    plt.plot(history.history['mae']) 
    plt.plot(history.history['val_mae']) 
    plt.title('Model mae') 
    plt.ylabel('Mae') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()

'''
Plot Forecast
'''
def plot_forecast(real, predict):
    plt.plot(history.history['mae']) 
    plt.plot(history.history['val_mae']) 
    plt.title('Model mae') 
    plt.ylabel('Mae') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()


###################################################################################################
###############################              MAIN              ####################################
###################################################################################################


'''
Vars
'''

nr_features = 13
timesteps = 24
multisteps = 1
epochs = 2
batch_size = 24
verbose = 2
patience = 15
n_splits = 2
now = datetime.datetime.now()

# Files
file_path = 'AvenidaAliados.csv'
#file_path = '../datasets/v1.0/AvenidaGustavoEiffel.csv'
#file_path = '../datasets/v1.0/RuaCondeVizela.csv'
#file_path = '../datasets/v1.0/RuaNovaAlfandega.csv'

# Load data
df = load_data(file_path)

# Prepare data
df_until_now, scaler, df_temp, drop_columns = prepare_data(df)

df_copy = df.copy()
df_all = df_copy.drop(columns=drop_columns)
df_all = fill_missing(df_all )

# Num features to use
print("Nr Features:", df_until_now.shape[1])
nr_features = df_until_now.shape[1]
columns = df_until_now.columns

# To supervisioned
X, Y = to_supervised(df_until_now, timesteps, multisteps, nr_features)

print("Shape X:", X.shape)
print("Shape Y:", Y.shape)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits)
cvscores = list()
split_num = 1
current_mae = 100
best_model = ''
for train_index, test_index in tscv.split(X):
    print(10*'-' + ' Begin Time Series Split Nº' + str(split_num) + ' ' + 10*'-')
    # Get values form time series split
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Create model
    model = build_model(timesteps, nr_features, multisteps)
    
    # Experiment the model
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.5, patience=patience, min_lr=0.00005)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', min_delta=0, patience=patience+5, verbose=verbose, mode='auto')

    # Fit the model
    print('> Fit the model, please wait...')
    history = model.fit(x_train, y_train, 
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size, 
                        verbose=verbose, 
                        shuffle=False,
                        callbacks=[early_stop,lr])

    # Plot loss
    plot_loss(history, str(split_num))
    
    # Plot mae
    plot_mae(history, str(split_num))
    
    # Evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=verbose)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)

    # Save best model
    if scores[1] < current_mae:
      current_mae = scores[1]
      best_model = model
    
    split_num += 1
    print(50*'-')
    print('\n')
    
# Final score
print(best_model.summary())
tf.keras.utils.plot_model(best_model, 'best_model.png', show_shapes=True)
print('Best (mae): %.2f%%' % (current_mae*100))
print('Final score(mae): %.2f%% (+/- %.2f%%)\n' % (np.mean(cvscores), np.std(cvscores)))

# Forecast
predictions = forecast(best_model, X, timesteps, multisteps, scaler, columns, nr_features, df_all_scaled)
step = 1 
print('\n')
print(10*('-') + ' ' + file_path + ' ' + 10*('-'))
print('\n> Prediction for the next %d MultiSteps\n' % multisteps)
print('> speed_diff(max): %d' % df_temp['speed_diff'].max())
print('> speed_diff(min): %d\n' % df_temp['speed_diff'].min())
for pred in predictions:
   next_multi_step = datetime.datetime.today() + datetime.timedelta(hours=step)
   print('> MultiStep Nº%d (%d(horas) - %d(dia) - %d(mes)): %.2f (speed_diff)' % (step, next_multi_step.hour, next_multi_step.day, next_multi_step.month, pred))
   step += 1

# Plot results / forecast
next_hours = datetime.datetime.today() + datetime.timedelta(hours=multisteps)
index = str(next_hours.hour) + '/' +  str(next_hours.day) + '/' + str(next_hours.month)
df_all_until_now = df_all.loc[:index]

df_last_timesteps = df_all_until_now.tail(timesteps)
print(df_all_until_now)
