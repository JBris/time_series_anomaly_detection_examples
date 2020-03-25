#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset import load_data, filter_dataset, index_timestamps, normalize_data, do_PCA

def covariance_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def view_anomalies(X_train_PCA, X_test_PCA):
    #Mahalanobis Distance test
    data_train = np.array(X_train_PCA.values)
    data_test = np.array(X_test_PCA.values)
    cov_matrix, inv_cov_matrix  = covariance_matrix(data_train)
    mean_distr = data_train.mean(axis=0)
    dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
    dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    threshold = MD_threshold(dist_train, extreme = True)

    plt.figure()
    sns.distplot(np.square(dist_train),
                bins = 10, 
                kde= False);
    plt.xlim([0.0,15])

    #Plot data
    plt.figure()
    sns.distplot(dist_train,
                bins = 10, 
                kde= True, 
                color = 'green');
    plt.xlim([0.0,5])
    plt.xlabel('Mahalanobis dist')
    plt.show()

    return dist_test, dist_train, threshold

def validate_data(dist_test, dist_train, threshold, X_train_PCA):
    anomaly_train = pd.DataFrame()
    anomaly_train['Mob dist']= dist_train
    anomaly_train['Thresh'] = threshold
    # If Mob dist above threshold: Flag as anomaly
    anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
    anomaly_train.index = X_train_PCA.index
    anomaly = pd.DataFrame()
    anomaly['Mob dist']= dist_test
    anomaly['Thresh'] = threshold
    # If Mob dist above threshold: Flag as anomaly
    anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
    anomaly.index = X_test_PCA.index
    print(anomaly.head())
    anomaly_alldata = pd.concat([anomaly_train, anomaly])
    anomaly_alldata.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])
    plt.show()

df = load_data()
dataset_train, dataset_test = filter_dataset(df)
dataset_train, dataset_test  = index_timestamps(dataset_train, dataset_test)
X_train, X_test = normalize_data(dataset_train, dataset_test)
X_train_PCA, X_test_PCA = do_PCA(X_train, X_test)
dist_test, dist_train, threshold = view_anomalies(X_train_PCA, X_test_PCA)
validate_data(dist_test, dist_train, threshold, X_train_PCA)
