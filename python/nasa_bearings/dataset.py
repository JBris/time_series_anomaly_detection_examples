#!/usr/bin/env python

import os
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(dirname + '/data/merged_dataset_BearingTest_2.csv')

def filter_dataset(df):
    dataset_train =  df.loc[ (df['timestamp'] >= '2004-02-12 11:02:39') & (df['timestamp'] < '2004-02-13 23:52:39') ]
    dataset_test = df.loc[df['timestamp'] >= '2004-02-13 23:52:39' ]
    return dataset_train, dataset_test

def index_timestamps(dataset_train, dataset_test):
    dataset_train.set_index('timestamp')
    dataset_test.set_index('timestamp')
    dataset_train = dataset_train.drop('timestamp', axis=1)
    dataset_test = dataset_test.drop('timestamp', axis=1)
    return dataset_train, dataset_test

def normalize_data(dataset_train, dataset_test):
    scaler = preprocessing.MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(dataset_train), 
                            columns=dataset_train.columns, 
                            index=dataset_train.index)# Random shuffle training data
    X_train.sample(frac=1)

    X_test = pd.DataFrame(scaler.transform(dataset_test), 
                            columns=dataset_test.columns, 
                            index=dataset_test.index)
    return X_train, X_test

def do_PCA(X_train, X_test):
    pca = PCA(n_components=2, svd_solver= 'full')
    
    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index

    X_test_PCA = pca.transform(X_test)
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index
    return X_train_PCA, X_test_PCA
