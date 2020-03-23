#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data
from datetime import datetime

df = load_data()
df.plot(x='timestamp', y='value', figsize=(12,6))
plt.xlabel('Date time')

df_length = len(df['timestamp'])
ticks_to_use = df.index[::int(df_length/5)]
plt.xticks(ticks_to_use)
plt.ylabel('CPU Utilization')
plt.title('Time Series of EC2 CPU utilization by date time');

start_date = datetime.strptime('14-02-24 00:00:00', '%y-%m-%d %H:%M:%S')
end_date = datetime.strptime('14-02-24 23:59:59', '%y-%m-%d %H:%M:%S')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df=df.loc[df['timestamp'] > start_date ]
df=df.loc[df['timestamp'] < end_date ]

df.plot(x='timestamp', y='value', figsize=(12,6))
plt.xlabel('Date time')

# df_length = len(df['timestamp'])
# ticks_to_use = df.index[::int(df_length/5)]
# plt.xticks(ticks_to_use)
plt.ylabel('CPU Utilization')
plt.title('Time Series of EC2 CPU utilization by date time');

plt.show();
