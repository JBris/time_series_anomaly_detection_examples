#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_date_series, timeseries_to_supervised, difference, inverse_difference, scale, invert_scale
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error as mse

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def prepare_model(series):
    # Convert series into supervised learning problem
    X = series.values
    supervised = timeseries_to_supervised(X, 1)
    print("*** Supervised Learning ***")
    print(supervised.head())

    # Convert time series to stationary
    differenced = difference(series, 1)
    print("*** Stationary Data Set ***")
    print(differenced.head())
    # invert transform
    inverted = list()
    for i in range(len(differenced)):
        value = inverse_difference(series, differenced[i], len(series)-i)
        inverted.append(value)
    inverted = pd.Series(inverted)
    print(inverted.head())

    # Scale time series
    scaler, scaled_X = scale(series)
    scaled_series = pd.Series(scaled_X[:, 0])
    print("*** Scaled Time Series ***")
    print(scaled_series.head())
    # invert transform
    inverted_X = scaler.inverse_transform(scaled_X)
    inverted_series = pd.Series(inverted_X[:, 0])
    print(inverted_series.head())

def model(series):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit the model
    lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = []
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        # store forecast
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]
        print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

    # report performance
    rmse = sqrt(mse(raw_values[-12:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    plt.plot(raw_values[-12:])
    plt.plot(predictions)
    plt.show()

def multiple_repeats(series):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    
    # repeat experiment
    repeats = 30
    error_scores = []
    for r in range(repeats):
        # fit the model
        lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        lstm_model.predict(train_reshaped, batch_size=1)
        # walk-forward validation on the test data
        predictions = []
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
        # report performance
        rmse = sqrt(mse(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
    
    # summarize results
    results = pd.DataFrame()
    results['rmse'] = error_scores
    print(results.describe())
    results.boxplot()
    plt.show()

series = load_date_series()
# prepare_model(series)
# model(series)
multiple_repeats(series)