#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data, filter_dataset

def view(df):
    ddict = filter_dataset(df)
    ddict['dataset_train'].plot(figsize = (12,6))
    plt.show();

df = load_data()
view(df)
