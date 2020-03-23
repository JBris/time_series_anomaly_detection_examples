#!/usr/bin/env python

from dataset import load_data

def explore(df):
    print(df.head())
    print()
    df.info()
    print()
    print(df['timestamp'].describe())
    print()
    print('Min: ' + df['timestamp'].min())
    print('Max: ' + df['timestamp'].max())

df = load_data()
explore(df)


