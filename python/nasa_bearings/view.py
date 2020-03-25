#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data

    # 
    # 
    # df['timestamp_str'] = df['timestamp']
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df=df.loc[df['timestamp'] > start_date ]
    # df=df.loc[df['timestamp'] < end_date ]

def view(df):
    dataset_train = df.loc[ (df['timestamp'] >= '2004-02-12 11:02:39') & (df['timestamp'] < '2004-02-13 23:52:39') ]
    dataset_test = df.loc[df['timestamp'] >= '2004-02-13 23:52:39' ]
    dataset_train.plot(figsize = (12,6))
    plt.show();

df = load_data()
view(df)
