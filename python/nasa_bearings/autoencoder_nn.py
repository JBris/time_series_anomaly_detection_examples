#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow  
from dataset import load_data, filter_dataset, index_timestamps, normalize_data
from keras import regularizers
from keras.layers.core import Dense 
from keras.models import Model, Sequential, load_model
from numpy.random import seed

def create_model(X_train, X_test):
    seed(10)
    tensorflow.random.set_seed(10)
    act_func = 'elu'

    # Input layer:
    model=Sequential()
    # First hidden layer, connected to input vector X. 
    model.add(Dense(10,activation=act_func,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(X_train.shape[1],)
                )
            )

    model.add(Dense(2,activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(10,activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(X_train.shape[1],
                    kernel_initializer='glorot_uniform'))

    model.compile(loss='mse',optimizer='adam')

    # Train model for 100 epochs, batch size of 10: 
    NUM_EPOCHS=100
    BATCH_SIZE=10

    history=model.fit(np.array(X_train),np.array(X_train),
                    batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS,
                    validation_split=0.05,
                    verbose = 1)

    plt.plot(history.history['loss'],
            'b',
            label='Training loss')
    plt.plot(history.history['val_loss'],
            'r',
            label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.ylim([0,.1])
    plt.show()
    return model 

def view_anomalies(X_train, X_test, model):
    #Plot the distribution of calculated loss to identify threshold
    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred, 
                        columns=X_train.columns)
    X_pred.index = X_train.index

    scored = pd.DataFrame(index=X_train.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    plt.figure()
    sns.distplot(scored['Loss_mae'],
                bins = 10, 
                kde= True,
                color = 'blue');
    plt.xlim([0.0,.5])
    plt.show()

    #Compare output to threshold 
    X_pred = model.predict(np.array(X_test))
    X_pred = pd.DataFrame(X_pred, 
                        columns=X_test.columns)
    X_pred.index = X_test.index

    scored = pd.DataFrame(index=X_test.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
    scored['Threshold'] = 0.3
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    print(scored.head())

    #Merge results into single dataframe and plot results
    X_pred_train = model.predict(np.array(X_train))
    X_pred_train = pd.DataFrame(X_pred_train, 
                        columns=X_train.columns)
    X_pred_train.index = X_train.index

    scored_train = pd.DataFrame(index=X_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
    scored_train['Threshold'] = 0.3
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    scored = pd.concat([scored_train, scored])

    scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])
    plt.show()

df = load_data()
dataset_train, dataset_test = filter_dataset(df)
dataset_train, dataset_test  = index_timestamps(dataset_train, dataset_test)
X_train, X_test = normalize_data(dataset_train, dataset_test)
model = create_model(X_train, X_test)
view_anomalies(X_train, X_test, model)
