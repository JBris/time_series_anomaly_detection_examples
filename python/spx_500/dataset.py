#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(dirname + '/data/spx.csv', index_col='date')

def write_timestamps():
    dirname = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(dirname + '/data/raw.csv', parse_dates=['date'])
    return df.to_csv(dirname + '/data/spx.csv', index=False)

def preprocess_data(df):
    #Get 95% of data set
    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train.shape, test.shape)
    #Scale dataset
    scaler = StandardScaler()
    scaler = scaler.fit(train[['close']])

    ct = ColumnTransformer([
        ('standardscaler', scaler, ['close'])
    ], remainder='passthrough')

    train['close'] = ct.fit_transform(train)
    test['close'] = ct.fit_transform(test)
    print(train.head())
    print(test.head())

    return train, test

def create_dataset(X, y, time_steps=1):
    #Split data set into subsequences
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)