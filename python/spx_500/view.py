#!/usr/bin/env python

import matplotlib.pyplot as plt
from dataset import load_data

def view(df):
    print(df.head())
    df.plot()
    plt.show()
    
df = load_data()
view(df)