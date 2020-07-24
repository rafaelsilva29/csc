#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:00:55 2020

@author: Grupo3 - CSC - 19/20
"""

import pandas as pd
import numpy as np
import talos
from talos.utils import hidden_layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.model_selection import TimeSeriesSplit
from numpy import nan
from numpy import isnan
from numpy import array


#tf.random.set_seed(91195003)
#np.random.seed(91195003)
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
    df_temp = fill_missing(data)
    print(df_temp.isnull().sum())
    
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
        end_prev = end_timesteps + 1
        if end_prev < data_size:
            X.append(data[curr_pos:end_timesteps,:])
            y.append(data[begin_prev:end_prev,5])
            
    X = np.reshape(np.array(X), (len(X), timesteps, nr_features))
    y = np.reshape(np.array(y), (len(y), 1))
        
    return X, y


'''
Tunning
'''
def tunnig(x_train, y_train, x_val, y_val, params):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(params['first_neuron'], 
                                    return_sequences=True,
                                    input_shape=(x_train.shape[1], x_train.shape[2]),
                                    activation=params['activation']))

    model.add(tf.keras.layers.LSTM(params['first_neuron'], 
                                    return_sequences=False,
                                    dropout=params['dropout'],
                                    activation=params['activation']))

    model.add(tf.keras.layers.Dense(params['first_neuron'],
                                    activation=params['activation']))
    model.add(tf.keras.layers.Dropout(params['dropout']))

    model.add(tf.keras.layers.Dense(1, activation=params['last_activation']))

    model.compile(
            loss=params['losses'],
            optimizer=params['optimizer'],
            metrics=['mae']
        )

    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)

    return history, model
