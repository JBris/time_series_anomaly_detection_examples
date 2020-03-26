#!/usr/bin/env python

from dataset import load_date_series, split_date_series
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import matplotlib.pyplot as plt

def model(train, test):
	# walk-forward validation
	history = [x for x in train]
	predictions = []
	# make prediction
	for i in range(len(test)):
		predictions.append(history[-1])
		# observation
		history.append(test[i])
	rmse = sqrt(mse(test, predictions))
	print('RMSE: %.3f' % rmse)
	# line plot of observed vs predicted
	plt.plot(test)
	plt.plot(predictions)
	plt.show()

series = load_date_series()
train, test = split_date_series(series)
model(train, test)
