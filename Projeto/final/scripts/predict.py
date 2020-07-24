#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:00:55 2020

@author: Grupo3 - CSC - 19/20
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

    # Make dataset numeric
    data = data.astype('float32')

    # Missing Value
    data_temp = fill_missing(data)
    #data_temp = data_temp.interpolate(method='linear', limit_direction='forward', axis=0)
    #print(data_temp.isnull().sum())

    # Normalized data
    scaler = MinMaxScaler(feature_range=norm_range)
    data_scaled = scaler.fit_transform(data_temp.values)
    new_data = pd.DataFrame(data_scaled,columns=data.columns,index=data.index)
    
    # Data until now
    index = str(now.hour) + '/' +  str(now.day) + '/' + str(now.month)
    df_until_now = new_data.loc[:index]

    return df_until_now, scaler, columns, new_data


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
        end_prev = end_timesteps + 1
        if end_prev < data_size:
            X.append(data[curr_pos:end_timesteps,:])
            y.append(data[begin_prev:end_prev,5])
            
    X = np.reshape(np.array(X), (len(X), timesteps, nr_features))
    y = np.reshape(np.array(y), (len(y), 1))
        
    return X, y


'''
Buid the model 
'''
def build_model(timesteps, features, multisteps, n_neurons=32, activation='tanh', dropout_rate=0.4):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(int(n_neurons),
                                   activation=activation,
                                   input_shape=(timesteps, features),
                                   return_sequences=True))

    model.add(tf.keras.layers.LSTM(int(n_neurons/2),
                                   activation=activation,
                                   dropout=dropout_rate,
                                   return_sequences=False))

    model.add(tf.keras.layers.Dense(int(n_neurons), activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='tanh'))

    model.compile(
            loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.Nadam(0.001),
            metrics=['mae']
        )
    
    return model


'''
Forecast
'''
def forecast(best_model, X, timesteps, multisteps, scaler, columns, nr_features, df_scaled):
    input_seq = X[-1:]
    inp = input_seq
    predictions = list()
    for step in range(1,multisteps+1):
        yhat = best_model.predict(inp, verbose=verbose)[0]
        yhat_inversed, inp = predict_multistep(yhat, scaler, columns, inp, timesteps, step, df_scaled)
        predictions.append(yhat_inversed)

    return predictions


'''
Predict multistep
'''
def predict_multistep(yhat, scaler, columns, inp, timesteps, step, df_scaled):
    # Save prediction(speed_diff) on data frame
    df_aux = pd.DataFrame(np.nan, index=range(0,1), columns=columns)
    df_aux['speed_diff'] = yhat[0]
    
    # Inverse scale of prediction
    yhat_inversed = scaler.inverse_transform(df_aux.values)[0][5]
    yhat_inversed = round(yhat_inversed, 2)
    
    # Concatenate data from lasts timestep and prediction, save on data frame
    inp = np.concatenate((inp, np.array(df_aux.values)[:,None]), axis=1)
    df_temp = pd.DataFrame(inp[0], index=range(0,timesteps+1), columns=columns)

    # Fill last multi_step with values df_scaled and prediction speed_diff    
    next_multi_step = datetime.datetime.today() + datetime.timedelta(hours=step)
    index = str(next_multi_step.hour) + '/' +  str(next_multi_step.day) + '/' + str(next_multi_step.month)
    df_temp.iloc[-1:] = df_scaled.loc[[index]].values
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
def plot_forecast(index, real, predictions, multisteps, title):
    plt.figure(figsize=(48, 10))
    plt.plot(index, real, color='green', label='Real')
    plt.plot(range(len(real)-multisteps, len(real)+len(predictions)-multisteps), predictions, color='red', label='Prediction') 
    plt.title('Forecast - ' + title) 
    plt.ylabel('Speed_Diff') 
    plt.xlabel('Index') 
    plt.legend(['Real (last 48h)', 'Predictions (next ' + str(multisteps) + 'h)'], loc='upper left')
    plt.savefig('forecast.png')
    plt.show()


###################################################################################################
###############################              MAIN              ####################################
###################################################################################################


'''
Vars
'''

nr_features = 13
timesteps = 24
multisteps = 24
epochs = 100
batch_size = 32
verbose = 0
patience = 10
n_splits = 5
now = datetime.datetime.now()

# Files
#file_path = '../datasets/AvenidaAliados.csv'
#file_path = '../datasets/AvenidaGustavoEiffel.csv'
file_path = '../datasets/RuaCondeVizela.csv'
#file_path = '../datasets/RuaNovaAlfandega.csv'

# Load data
df = load_data(file_path)

# Prepare data
df_until_now, scaler, drop_columns, df_scaled = prepare_data(df)

df_copy = df.copy()
df_complete = df_copy.drop(columns=drop_columns)
df_complete = fill_missing(df_complete)

# Num features to use
print("Nr Features:", df_until_now.shape[1])
nr_features = df_until_now.shape[1]
columns = df_until_now.columns

# To supervisioned
X, Y = to_supervised(df_scaled, timesteps, multisteps, nr_features)

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
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', min_delta=0, patience=25, verbose=verbose, mode='auto')

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
    print(49*'-')
    print('\n')

 
# Final score
print(best_model.summary())
tf.keras.utils.plot_model(best_model, 'best_model.png', show_shapes=True)
print('Best (mae): %.2f%%' % (current_mae*100))
print('Final score(mae): %.2f%% (+/- %.2f%%)\n' % (np.mean(cvscores), np.std(cvscores)))


# Forecast
X_now, y_now = to_supervised(df_until_now, timesteps, multisteps, nr_features)
predictions = forecast(best_model, X_now, timesteps, multisteps, scaler, columns, nr_features, df_scaled)
step = 1 
print('\n')
print(10*('-') + ' ' + file_path + ' ' + 10*('-'))
df_until_now_normal = scaler.inverse_transform(df_until_now)
aux_df = pd.DataFrame(df_until_now_normal,columns=df_until_now.columns,index=df_until_now.index)
print('\n> Prediction for the next %d MultiSteps\n' % multisteps)
print('> speed_diff(max): %d' % aux_df['speed_diff'].max())
print('> speed_diff(min): %d\n' % aux_df['speed_diff'].min())
for pred in predictions:
    next_multi_step = datetime.datetime.today() + datetime.timedelta(hours=step)
    print('> MultiStep Nº%d (%d(horas) - %d(dia) - %d(mes)): %.2f (speed_diff)' % (step, next_multi_step.hour, next_multi_step.day, next_multi_step.month, pred))
    step += 1


# Plot results / forecast
next_hours = datetime.datetime.today() + datetime.timedelta(hours=multisteps)
index = str(next_hours.hour) + '/' +  str(next_hours.day) + '/' + str(next_hours.month)
df_complete_until_now = df_complete.loc[:index]
df_last_timesteps = df_complete_until_now.tail(timesteps*2)
plot_forecast(df_last_timesteps.index.values, df_last_timesteps['speed_diff'].values, predictions, multisteps, file_path)


# Evalute model
df_aux = pd.DataFrame(np.nan, index=range(0,1), columns=columns)
df_aux['speed_diff'] = current_mae
real_error = scaler.inverse_transform(df_aux.values)[0][5]
print("Real error (Km/h): %.2fKm/h\n"  % round(real_error,2))

df_last_timesteps_aux = df_complete_until_now.tail(timesteps)['speed_diff'].values
errors = []
for step in range(len(predictions)):
    error = abs(predictions[step]-df_last_timesteps_aux[step])
    errors.append(error)
print('Final error (Km/h): %.2fKm/h (+/- %.2fKm/h)\n' % (np.mean(errors), np.std(errors)))
