#!/usr/bin/env python

# Source: https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/
# Author: Jason Brownlee - https://machinelearningmastery.com/author/jasonb/
# Data: https://www.kaggle.com/c/dsg-hackathon/data

import os
import pandas as pd
import numpy as np
import sys

dirname = os.path.dirname(os.path.realpath(__file__))
if len(sys.argv) == 2: 
    arg = sys.argv[1]
else: 
    arg = None

# load dataset
def load_data():
    return pd.read_csv(dirname + '/data/dsg/TrainingData.csv')

# split the dataset by 'chunkID', return a dict of id to rows
def to_chunks(values, chunk_ix=1):
	chunks = {}
	# get the unique chunk ids
	chunk_ids = np.unique(values[:, chunk_ix])
	# group rows by chunk id
	for chunk_id in chunk_ids:
		selection = values[:, chunk_ix] == chunk_id
		chunks[chunk_id] = values[selection, :]
	return chunks

# return a list of relative forecast lead times
def get_lead_times():
	return [1, 2 ,3, 4, 5, 10, 17, 24, 48, 72]
	
# split each chunk into train/test sets
def split_train_test(chunks, row_in_chunk_ix=2):
	train, test = [], []
	# first 5 days of hourly observations for train
	cut_point = 5 * 24
	# enumerate chunks
	for k,rows in chunks.items():
		# split chunk rows by 'position_within_chunk'
		train_rows = rows[rows[:,row_in_chunk_ix] <= cut_point, :]
		test_rows = rows[rows[:,row_in_chunk_ix] > cut_point, :]
		if len(train_rows) == 0 or len(test_rows) == 0:
			print('>dropping chunk=%d: train=%s, test=%s' % (k, train_rows.shape, test_rows.shape))
			continue
		# store with chunk id, position in chunk, hour and all targets
		indices = [1,2,5] + [x for x in range(56,train_rows.shape[1])]
		train.append(train_rows[:, indices])
		test.append(test_rows[:, indices])
	return train, test

# convert the rows in a test chunk to forecasts
def to_forecasts(test_chunks, row_in_chunk_ix=1):
	# get lead times
	lead_times = get_lead_times()
	# first 5 days of hourly observations for train
	cut_point = 5 * 24
	forecasts = []
	# enumerate each chunk
	for rows in test_chunks:
		chunk_id = rows[0, 0]
		# enumerate each lead time
		for tau in lead_times:
			# determine the row in chunk we want for the lead time
			offset = cut_point + tau
			# retrieve data for the lead time using row number in chunk
			row_for_tau = rows[rows[:,row_in_chunk_ix]==offset, :]
			# check if we have data
			if len(row_for_tau) == 0:
				# create a mock row [chunk, position, hour] + [nan...]
				row = [chunk_id, offset, np.nan] + [np.nan for _ in range(39)]
				forecasts.append(row)
			else:
				# store the forecast row
				forecasts.append(row_for_tau[0])
	return np.array(forecasts)

def save_data_split(train_rows, test_rows):
    if arg == "test" or arg == "t": 
        np.savetxt( dirname + '/data/dsg/naive_train.csv', train_rows, delimiter=',')
        np.savetxt( dirname + '/data/dsg/naive_test.csv', test_rows, delimiter=',')


df = load_data()
# group data by chunks
values = df.values
chunks = to_chunks(values)
print('Total Chunks: %d' % len(chunks))
# split into train/test
train, test = split_train_test(chunks)
# flatten training chunks to rows
train_rows = np.array([row for rows in train for row in rows])
# print(train_rows.shape)
print('Train Rows: %s' % str(train_rows.shape))
# reduce train to forecast lead times only
test_rows = to_forecasts(test)
print('Test Rows: %s' % str(test_rows.shape))
save_data_split(train_rows, test_rows)

