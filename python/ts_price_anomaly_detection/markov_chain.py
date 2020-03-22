import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import load_data, filter_dataset, reindex_data
from sklearn.covariance import EllipticEnvelope
from pyemma import msm

# train markov model to get transition matrix
def getTransitionMatrix (df):
    df = np.array(df)
    model = msm.estimate_markov_model(df, 1)
    return model.transition_matrix

# return the success probability of the state change 
def successProbabilityMetric(state1, state2, transition_matrix):
    proba = 0
    for k in range(0,len(transition_matrix)):
        if (k != (state2-1)):
            proba += transition_matrix[state1-1][k]
    return 1-proba

# return the success probability of the whole sequence
def sucessScore(sequence, transition_matrix):
    proba = 0 
    for i in range(1,len(sequence)):
        if(i == 1):
            proba = successProbabilityMetric(sequence[i-1], sequence[i], transition_matrix)
        else:
            proba = proba*successProbabilityMetric(sequence[i-1], sequence[i], transition_matrix)
    return proba

# return if the sequence is an anomaly considering a threshold
def anomalyElement(sequence, threshold, transition_matrix):
    if (sucessScore(sequence, transition_matrix) > threshold):
        return 0
    else:
        return 1

# return a dataframe containing anomaly result for the whole dataset 
# choosing a sliding windows size (size of sequence to evaluate) and a threshold
def markovAnomaly(df, windows_size, threshold):
    transition_matrix = getTransitionMatrix(df)
    real_threshold = threshold**windows_size
    df_anomaly = []
    for j in range(0, len(df)):
        if (j < windows_size):
            df_anomaly.append(0)
        else:
            sequence = df[j-windows_size:j]
            sequence = sequence.reset_index(drop=True)
            df_anomaly.append(anomalyElement(sequence, real_threshold, transition_matrix))
    return df_anomaly
    
def view_anomalies(df):
    data = reindex_data(df)
    df.index = data.index
    # definition of the different state
    x1 = (df['price_usd'] <=55).astype(int)
    x2= ((df['price_usd'] > 55) & (df['price_usd']<=200)).astype(int)
    x3 = ((df['price_usd'] > 200) & (df['price_usd']<=300)).astype(int)
    x4 = ((df['price_usd'] > 300) & (df['price_usd']<=400)).astype(int)
    x5 = (df['price_usd'] >400).astype(int)
    df_mm = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5

    # getting the anomaly labels for our dataset (evaluating sequence of 5 values and anomaly = less than 20% probable)
    df_anomaly = markovAnomaly(df_mm, 5, 0.20)
    df_anomaly = pd.Series(df_anomaly)
    print(df_anomaly.value_counts())
    df['anomaly24'] = df_anomaly
    fig, ax = plt.subplots(figsize=(10, 6))

    df = df.sort_values('date_time')
    df['date_time_int'] = pd.to_datetime(df['date_time']).astype('int64')
    a = df.loc[df['anomaly24'] == 1, ('date_time_int', 'price_usd')] #anomaly
    ax.plot(df['date_time_int'], df['price_usd'], color='blue')
    ax.scatter(a['date_time_int'],a['price_usd'], color='red')

    a = df.loc[df['anomaly24'] == 0, 'price_usd']
    b = df.loc[df['anomaly24'] == 1, 'price_usd']

    fig, axs = plt.subplots(figsize=(16,6))
    axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])
    plt.show();

expedia = load_data()
df = filter_dataset(expedia)
view_anomalies(df)