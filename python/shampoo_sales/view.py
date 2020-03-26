#!/usr/bin/env python

from dataset import load_date_series
from matplotlib import pyplot as plt

def view(series):
    # summarize first few rows
    print(series.head())
    # line plot
    series.plot()
    plt.show()

series = load_date_series()
view(series)
