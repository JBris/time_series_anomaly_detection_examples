#!/usr/bin/env python

import os
import pandas as pd

def load_date_series():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(dirname + '/data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

def split_date_series(series):
    # split data into train and test
    X = series.values
    train, test = X[0:-12], X[-12:]
    return train, test
