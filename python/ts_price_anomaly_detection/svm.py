#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data, filter_dataset, reindex_data
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def view_anomalies(df):
    data = reindex_data(df)
    df.index = data.index

    data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)

    # train oneclassSVM 
    outliers_fraction = 0.01
    model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
    model.fit(data)
    df['anomaly3'] = pd.Series(model.predict(data))

    fig, ax = plt.subplots(figsize=(10,6))

    df = df.sort_values('date_time')
    df['date_time_int'] = pd.to_datetime(df['date_time']).astype('int64')
    a = df.loc[df['anomaly3'] == -1, ['date_time_int', 'price_usd']] #anomaly

    ax.plot(df['date_time_int'], df['price_usd'], color='blue', label='Normal')
    ax.scatter(a['date_time_int'],a['price_usd'], color='red', label = 'Anomaly')

    a = df.loc[df['anomaly3'] == 1, 'price_usd']
    b = df.loc[df['anomaly3'] == -1, 'price_usd']

    fig, axs = plt.subplots(figsize=(10,6))
    axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])
    plt.show();

expedia = load_data()
df = filter_dataset(expedia)
view_anomalies(df)
