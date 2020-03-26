#!/usr/bin/env python

import os
import pandas as pd
from datetime import datetime

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(dirname + '/data/spx.csv', index_col='date')

def write_timestamps():
    dirname = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(dirname + '/data/raw.csv', parse_dates=['date'])
    return df.to_csv(dirname + '/data/spx.csv', index=False)
