#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:17:21 2020

@author: Grupo3 - CSC - 19/20
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
import xgboost as xgb
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
    df = pd.read_csv(file, encoding='utf-8', index_col='row ID', header=0)
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
    # Make dataset numeric
    df = df.astype('float32')
    
    # Missing Values
    df = fill_missing(df)
    df= df.interpolate(method='linear', limit_direction='forward', axis=0)
    #print(df.isnull().sum())
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=norm_range)
    normalized_df = scaler.fit_transform(df.values)
    new_df = pd.DataFrame(normalized_df,columns=df.columns,index=df.index)
    
    # Target & Features
    columns = ['speed_diff','time_diff']
    y = new_df['speed_diff']
    X = new_df.drop(columns=columns)
    
    return X, y, scaler


###################################################################################################
###############################              MAIN              ####################################
###################################################################################################


# Files
file = '../datasets/AvenidaAliados.csv'
#file = '../datasets/AvenidaGustavoEiffel.csv'
#file = '../datasets/RuaCondeVizela.csv'
#file = '../datasets/RuaNovaAlfandega.csv'

# Load data
df = load_data(file)

# Prepare data
X, y, scaler = prepare_data(df)


###################################################################################################
############### Random Search #####################################################################

logistic = DecisionTreeRegressor()
param_dist = {
                  "max_depth": [3, None],
                  "max_features": randint(1, 17),
                  "min_samples_leaf": randint(1, 20)
              }
clf = RandomizedSearchCV(logistic, param_dist, random_state=0)

search = clf.fit(X, y)
print("Features:",search.best_params_)

model = DecisionTreeRegressor()
rfe = RFE(model,n_features_to_select=13, step=10)
fit = rfe.fit(X, y)

print('\n########### Selected Features ###########')
print("Feature Ranking: %s" % (fit.ranking_))
print("Num Features: %s" % (fit.n_features_))

height = fit.support_
bars = X.columns
y_pos = np.arange(len(bars))
plt.figure(figsize=(5, 15))
plt.barh(y_pos, height)
plt.yticks(y_pos, bars)
plt.show()


###################################################################################################
################# XGBoost #########################################################################

xgb_params = {
    'eta': 0.05,
    'max_depth': 10,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X, y, feature_names=X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
remain_num = 99

fig, ax = plt.subplots(figsize=(10,18))
xgb.plot_importance(model, max_num_features=remain_num, height=0.8, ax=ax)
plt.show()
