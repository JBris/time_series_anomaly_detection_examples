#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import load_data, filter_dataset, reindex_data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def elbow_method(df):
    data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
    n_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
    scores = [kmeans[i].score(data) for i in range(len(kmeans))]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(n_cluster, scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show();

def generate_k_means(df):
    X = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
    X = X.reset_index(drop=True)
    km = KMeans(n_clusters=10)
    km.fit(X)
    km.predict(X)
    labels = km.labels_
    #Plotting
    fig = plt.figure(1, figsize=(7,7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2],
            c=labels.astype(np.float), edgecolor="k")
    ax.set_xlabel("price_usd")
    ax.set_ylabel("srch_booking_window")
    ax.set_zlabel("srch_saturday_night_bool")
    plt.title("K Means", fontsize=14);
    plt.show();

def do_PCA(df):
    data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
    X = data.values
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key = lambda x: x[0], reverse= True)
    tot = sum(eig_vals)
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
    cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(var_exp)), var_exp, alpha=0.3, align='center', label='individual explained variance', color = 'g')
    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show();

def getDistanceByPoint(data, model):
    distance = pd.Series(dtype='float64')
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.at[i] = np.linalg.norm(Xa-Xb)
    return distance

def view_anomalies(df):
    data = reindex_data(df)
    df.index = data.index
    n_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
    df['cluster'] = kmeans[9].predict(data)
    df['principal_feature1'] = data[0]
    df['principal_feature2'] = data[1]
    df['cluster'].value_counts()

    outliers_fraction = 0.01
    # get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
    distance = getDistanceByPoint(data, kmeans[9])
    number_of_outliers = int(outliers_fraction*len(distance))
    threshold = distance.nlargest(number_of_outliers).min()
    # anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
    df['anomaly1'] = (distance >= threshold).astype(int)

    # visualisation of anomaly with cluster view
    fig, ax = plt.subplots(figsize=(10,6))
    colors = {0:'blue', 1:'red'}
    ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["anomaly1"].apply(lambda x: colors[x]))
    plt.xlabel('principal feature1')
    plt.ylabel('principal feature2')

    df = df.sort_values('date_time')
    df['date_time_int'] = pd.to_datetime(df['date_time']).astype('int64')
    fig, ax = plt.subplots(figsize=(10,6))
    a = df.loc[df['anomaly1'] == 1, ['date_time_int', 'price_usd']] #anomaly

    ax.plot(df['date_time_int'], df['price_usd'], color='blue', label='Normal')
    ax.scatter(a['date_time_int'],a['price_usd'], color='red', label='Anomaly')
    plt.xlabel('Date Time Integer')
    plt.ylabel('price in USD')
    plt.legend()

    # visualisation of anomaly with avg price repartition
    a = df.loc[df['anomaly1'] == 0, 'price_usd']
    b = df.loc[df['anomaly1'] == 1, 'price_usd']
    fig, axs = plt.subplots(figsize=(10,6))
    axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])
    plt.show();

expedia = load_data()
df = filter_dataset(expedia)
#elbow_method(df)
#generate_k_means(df)
# do_PCA(df)
view_anomalies(df)