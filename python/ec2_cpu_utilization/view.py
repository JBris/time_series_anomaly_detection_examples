#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data, filter_data

def view(df):
    df.plot(x='timestamp', y='value', figsize=(12,6))
    plt.xlabel('Date time')

    df_length = len(df['timestamp'])
    ticks_to_use = df.index[::int(df_length/5)]
    plt.xticks(ticks_to_use)
    plt.ylabel('CPU Utilization')
    plt.title('Time Series of EC2 CPU utilization by date time');
    df = filter_data(df)
    df.plot(x='timestamp', y='value', figsize=(12,6))
    plt.xlabel('Date time')
    plt.ylabel('CPU Utilization')
    plt.title('Time Series of EC2 CPU utilization by date time');

    plt.show();

df = load_data()
view(df)
