#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data, filter_dataset, reindex_data
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def view_anomalies(df):
    data = reindex_data(df)
    df.index = data.index

    data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)

    # train isolation forest
    outliers_fraction = 0.01
    model =  IsolationForest(contamination=outliers_fraction)
    model.fit(data) 
    df['anomaly2'] = pd.Series(model.predict(data))

    # visualization
    fig, ax = plt.subplots(figsize=(10,6))
    
    df = df.sort_values('date_time')
    df['date_time_int'] = pd.to_datetime(df['date_time']).astype('int64')
    a = df.loc[df['anomaly2'] == -1, ['date_time_int', 'price_usd']] #anomaly

    ax.plot(df['date_time_int'], df['price_usd'], color='blue', label = 'Normal')
    ax.scatter(a['date_time_int'],a['price_usd'], color='red', label = 'Anomaly')
    plt.legend()

    df['anomaly2'].unique()
    # visualisation of anomaly with avg price repartition
    a = df.loc[df['anomaly2'] == 1, 'price_usd']
    b = df.loc[df['anomaly2'] == -1, 'price_usd']

    fig, axs = plt.subplots(figsize=(10,6))
    axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])

    plt.show();

expedia = load_data()
df = filter_dataset(expedia)
view_anomalies(df)