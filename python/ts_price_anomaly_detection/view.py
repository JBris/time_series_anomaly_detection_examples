#!/usr/bin/env python

import matplotlib.pyplot as plt
from dataset import load_data, filter_dataset

def view(df):
    df.plot(x='date_time', y='price_usd', figsize=(12,6))
    plt.xlabel('Date time')
    plt.ylabel('Price in USD')
    plt.title('Time Series of room price by date time of search');

    a = df.loc[df['srch_saturday_night_bool'] == 0, 'price_usd']
    b = df.loc[df['srch_saturday_night_bool'] == 1, 'price_usd']
    plt.figure(figsize=(10, 6))
    plt.hist(a, bins = 50, alpha=0.5, label='Search Non-Sat Night')
    plt.hist(b, bins = 50, alpha=0.5, label='Search Sat Night')
    plt.legend(loc='upper right')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.show();

expedia = load_data()
df = filter_dataset(expedia)
view(df)
