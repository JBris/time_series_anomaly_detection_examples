#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from dataset import load_data

def view(df):
    print(df.shape)
    print(df.head())
    df.iloc[8400:8400+7*48,:].plot(y='value', x='timestamp', figsize=(8,6))
    timeLags = np.arange(1,10*48*7)
    autoCorr = [df.value.autocorr(lag=dt) for dt in timeLags]

    plt.figure(figsize=(19,8))
    plt.plot(1.0/(48*7)*timeLags, autoCorr);
    plt.xlabel('time lag [weeks]'); 
    plt.ylabel('correlation coeff', fontsize=12);
    plt.show()

df = load_data()
view(df)