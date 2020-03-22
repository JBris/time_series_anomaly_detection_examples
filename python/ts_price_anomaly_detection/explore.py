#!/usr/bin/env python

from dataset import load_data, filter_dataset

def explore(df):
    df.info()
    print(df['price_usd'].describe())
    return df

expedia = load_data()
df = filter_dataset(expedia)
explore(df)