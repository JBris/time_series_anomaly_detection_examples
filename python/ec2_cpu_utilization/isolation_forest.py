#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data, filter_data
from datetime import datetime
from sklearn.ensemble import IsolationForest

def iqr_bounds(scores,k=1.5):
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower_bound=(q1 - k * iqr)
    upper_bound=(q3 + k * iqr)
    print("Lower bound:{} \nUpper bound:{}".format(lower_bound,upper_bound))
    return lower_bound,upper_bound

def view_anomalies(df):

    #Fixed contamination value
    clf=IsolationForest(n_estimators=10, max_samples='auto', contamination=float(.04), \
                            max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(df[['value']])
    df['scores']=clf.decision_function(df[['value']])
    df['anomaly']=clf.predict(df[['value']])
    df.head()
    df.loc[df['anomaly'] == 1,'anomaly'] = 0
    df.loc[df['anomaly'] == -1,'anomaly'] = -1
    print(df['anomaly'].value_counts())

    # visualization
    fig, ax = plt.subplots(figsize=(10,6))
    a = df.loc[df['anomaly'] == -1, ['timestamp', 'value']] #anomaly
    ax.plot(df['timestamp'], df['value'], color='blue', label = 'Normal')
    ax.scatter(a['timestamp'],a['value'], color='red', label = 'Anomaly')
    plt.legend()
    plt.show();

    df['scores'].hist()
    plt.show();

    #IQR-based  
    print()
    lower_bound,upper_bound=iqr_bounds(df['scores'],k=2)

    df['anomaly']=0
    df['anomaly']=(df['scores'] < lower_bound) |(df['scores'] > upper_bound)
    df['anomaly']=df['anomaly'].astype(int)
    fig, ax = plt.subplots(figsize=(10,6))
    a = df.loc[df['anomaly'] == 1, ['timestamp', 'value']] #anomaly
    ax.plot(df['timestamp'], df['value'], color='blue', label = 'Normal')
    ax.scatter(a['timestamp'],a['value'], color='red', label = 'Anomaly')
    plt.title('IQR-based');
    plt.legend()
    plt.show();

    print("Percentage of anomalies in data: {:.2f}".format((len(df.loc[df['anomaly']==1])/len(df))*100))
    return df

def validate_model(full_df):
    start_date = datetime.strptime('14-02-17 00:00:00', '%y-%m-%d %H:%M:%S')
    end_date = datetime.strptime('14-02-17 23:59:59', '%y-%m-%d %H:%M:%S')

    df=full_df.loc[ (full_df['timestamp'] > start_date) & (full_df['timestamp'] < end_date) ]
    # Using graph_objects
    df.plot(x='timestamp', y='value', figsize=(12,6))
    plt.xlabel('Date time')
    plt.ylabel('CPU Utilization')
    plt.title('Distribution of Validation Data');
    plt.show();

    #Fixed contamination value
    clf=IsolationForest(n_estimators=10, max_samples='auto', contamination=float(.04), \
                            max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(df[['value']])
    df['scores']=clf.decision_function(df[['value']])
    df['anomaly']=clf.predict(df[['value']])
    df.loc[df['anomaly'] == 1,'anomaly'] = 0
    df.loc[df['anomaly'] == -1,'anomaly'] = -1
    
    fig, ax = plt.subplots(figsize=(10,6))
    a = df.loc[df['anomaly'] == -1, ['timestamp', 'value']] #anomaly
    ax.plot(df['timestamp'], df['value'], color='blue', label = 'Normal')
    ax.scatter(a['timestamp'],a['value'], color='red', label = 'Anomaly')
    plt.legend()
    plt.show();

    print("Percentage of anomalies in data: {:.2f}".format((len(df.loc[df['anomaly']==1])/len(df))*100))
    df['scores'].hist()
    plt.show();

    #IQR-based  
    lower_bound,upper_bound=iqr_bounds(df['scores'],k=2)
    df['anomaly']=0
    df['anomaly']=(df['scores'] < lower_bound) |(df['scores'] > upper_bound)
    df['anomaly']=df['anomaly'].astype(int)

    fig, ax = plt.subplots(figsize=(10,6))
    a = df.loc[df['anomaly'] == 1, ['timestamp', 'value']] #anomaly
    ax.plot(df['timestamp'], df['value'], color='blue', label = 'Normal')
    ax.scatter(a['timestamp'],a['value'], color='red', label = 'Anomaly')
    plt.title('IQR-based');
    plt.legend()
    plt.show();
    print("Percentage of anomalies in data: {:.2f}".format((len(df.loc[df['anomaly']==1])/len(df))*100))
    
full_df = load_data()
df = filter_data(full_df)
df = view_anomalies(df)
validate_model(full_df)
