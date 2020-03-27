#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import tqdm
from dataset import load_data
from sklearn.metrics import mean_squared_log_error
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

### CREATE GENERATOR FOR LSTM ###
def gen_index(id_df, seq_length, seq_cols):
    data_matrix =  id_df[seq_cols]
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):     
        yield data_matrix[ stop-seq_length : stop ].values.reshape((-1,len(seq_cols)))

### DEFINE QUANTILE LOSS ###
def q_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def prepare_data(df):
    ### CREATE WEEKDAY FEATURE AND COMPUTE THE MEAN FOR WEEKDAYS AT EVERY HOURS ###
    df['weekday'] = df.timestamp.dt.weekday
    df['weekday_hour'] = df.weekday.astype(str) +' '+ df.H.astype(str)
    df['m_weekday'] = df.weekday_hour.replace(df[:5000].groupby('weekday_hour')['value'].mean().to_dict())

    sequence_length = 48
    ### CREATE AND STANDARDIZE DATA FOR LSTM ### 
    cnt, mean = [], []
    for sequence in gen_index(df, sequence_length, ['value']):
        cnt.append(sequence)
        
    for sequence in gen_index(df, sequence_length, ['m_weekday']):
        mean.append(sequence)

    cnt, mean = np.log(cnt), np.log(mean)
    cnt = cnt - mean
    print(cnt.shape)

    ## CREATE AND STANDARDIZE LABEL FOR LSTM ###
    init = df.m_weekday[sequence_length:].apply(np.log).values
    label = df.value[sequence_length:].apply(np.log).values - init
    print(label.shape)

    ### TRAIN TEST SPLIT ###
    X_train, X_test = cnt[:5000], cnt[5000:]
    y_train, y_test = label[:5000], label[5000:]
    train_date, test_date = df.timestamp.values[sequence_length:5000+sequence_length], df.timestamp.values[5000+sequence_length:]

    return X_train, X_test, y_train, y_test, train_date, test_date, init

def model(X_train, X_test, y_train, y_test, train_date, test_date):
    tf.random.set_seed(33)
    os.environ['PYTHONHASHSEED'] = str(33)
    np.random.seed(33)
    random.seed(33)

    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, 
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), 
        config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)

    ### CREATE MODEL ###
    losses = [lambda y,f: q_loss(0.1,y,f), lambda y,f: q_loss(0.5,y,f), lambda y,f: q_loss(0.9,y,f)]
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(inputs, training = True)
    lstm = Bidirectional(LSTM(16, return_sequences=False, dropout=0.3))(lstm, training = True)
    dense = Dense(50)(lstm)
    out10 = Dense(1)(dense)
    out50 = Dense(1)(dense)
    out90 = Dense(1)(dense)
    model = Model(inputs, [out10,out50,out90])

    epochs = 50
    model.compile(loss=losses, optimizer='adam', loss_weights = [0.3,0.3,0.3])
    model.fit(X_train, [y_train,y_train,y_train], epochs=epochs, batch_size=128, verbose=2)
    return model

def evaluate_model(model, X_test, y_test, test_date, init):
    ### QUANTILEs BOOTSTRAPPING ###
    pred_10, pred_50, pred_90 = [], [], []
    NN = K.function([model.layers[0].input], 
                    [model.layers[-3].output,model.layers[-2].output,model.layers[-1].output])

    iterations = 100
    for i in tqdm.tqdm(range(0, iterations)):
        predd = NN([X_test, 0.5])
        pred_10.append(predd[0])
        pred_50.append(predd[1])
        pred_90.append(predd[2])

    pred_10 = np.asarray(pred_10)[:,:,0] 
    pred_50 = np.asarray(pred_50)[:,:,0]
    pred_90 = np.asarray(pred_90)[:,:,0]

    ### REVERSE TRANSFORM PREDICTIONS ###
    pred_90_m = np.exp(np.quantile(pred_90,0.9,axis=0) + init[5000:])
    pred_50_m = np.exp(pred_50.mean(axis=0) + init[5000:])
    pred_10_m = np.exp(np.quantile(pred_10,0.1,axis=0) + init[5000:])

    ### EVALUATION METRIC ###
    mean_squared_log_error(np.exp(y_test + init[5000:]), pred_50_m)

    ### PLOT QUANTILE PREDICTIONS ###
    plt.figure(figsize=(16,8))
    plt.plot(test_date, pred_90_m, color='cyan')
    plt.plot(test_date, pred_50_m, color='blue')
    plt.plot(test_date, pred_10_m, color='green')
    plt.show();

    # ### CROSSOVER CHECK ###
    plt.scatter(np.where(np.logical_or(pred_50_m>pred_90_m, pred_50_m<pred_10_m))[0], 
                pred_50_m[np.logical_or(pred_50_m>pred_90_m, pred_50_m<pred_10_m)], c='red', s=50)

    ### PLOT UNCERTAINTY INTERVAL LENGHT WITH REAL DATA ###
    plt.figure(figsize=(16,8))
    plt.plot(test_date, np.exp(y_test + init[5000:]), color='red', alpha=0.4)
    plt.scatter(test_date, pred_90_m - pred_10_m)
    plt.show();

df = load_data()
X_train, X_test, y_train, y_test, train_date, test_date, init = prepare_data(df)
model = model(X_train, X_test, y_train, y_test, train_date, test_date)
evaluate_model(model, X_test, y_test, test_date, init)