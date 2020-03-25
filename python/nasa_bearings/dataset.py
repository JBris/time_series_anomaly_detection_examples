#!/usr/bin/env python

import os
import pandas as pd

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(dirname + '/data/merged_dataset_BearingTest_2.csv')
