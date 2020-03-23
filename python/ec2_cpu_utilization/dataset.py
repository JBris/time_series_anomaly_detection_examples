#!/usr/bin/env python

import os
import pandas as pd
from datetime import datetime

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(dirname + '/data/ec2_cpu_utilization_5f5533.csv')

def filter_data(df):
    start_date = datetime.strptime('14-02-24 00:00:00', '%y-%m-%d %H:%M:%S')
    end_date = datetime.strptime('14-02-24 23:59:59', '%y-%m-%d %H:%M:%S')
    df['timestamp_str'] = df['timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df=df.loc[df['timestamp'] > start_date ]
    df=df.loc[df['timestamp'] < end_date ]
    return df