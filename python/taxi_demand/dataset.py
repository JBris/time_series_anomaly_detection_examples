#!/usr/bin/env python

import os
import pandas as pd

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(dirname + '/data/nyc_taxi.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['yr'] = df.timestamp.dt.year
    df['mt'] = df.timestamp.dt.month
    df['d'] = df.timestamp.dt.day
    df['H'] = df.timestamp.dt.hour
    return df