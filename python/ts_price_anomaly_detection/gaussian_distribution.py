import matplotlib.pyplot as plt
import pandas as pd
from dataset import load_data, filter_dataset, reindex_data
from sklearn.covariance import EllipticEnvelope

def view_anomalies(df):
    data = reindex_data(df)
    df.index = data.index

    df_class0 = df.loc[df['srch_saturday_night_bool'] == 0, 'price_usd']
    df_class1 = df.loc[df['srch_saturday_night_bool'] == 1, 'price_usd']

    fig, axs = plt.subplots(1,2)
    df_class0.hist(ax=axs[0], bins=30)
    df_class1.hist(ax=axs[1], bins=30);

    outliers_fraction = 0.01
    envelope =  EllipticEnvelope(contamination = outliers_fraction) 
    X_train = df_class0.values.reshape(-1,1)
    envelope.fit(X_train)
    df_class0 = pd.DataFrame(df_class0)
    df_class0['deviation'] = envelope.decision_function(X_train)
    df_class0['anomaly'] = envelope.predict(X_train)

    envelope =  EllipticEnvelope(contamination = outliers_fraction) 
    X_train = df_class1.values.reshape(-1,1)
    envelope.fit(X_train)
    df_class1 = pd.DataFrame(df_class1)
    df_class1['deviation'] = envelope.decision_function(X_train)
    df_class1['anomaly'] = envelope.predict(X_train)

    # plot the price repartition by categories with anomalies
    a0 = df_class0.loc[df_class0['anomaly'] == 1, 'price_usd']
    b0 = df_class0.loc[df_class0['anomaly'] == -1, 'price_usd']

    a2 = df_class1.loc[df_class1['anomaly'] == 1, 'price_usd']
    b2 = df_class1.loc[df_class1['anomaly'] == -1, 'price_usd']

    fig, axs = plt.subplots(1,2)
    axs[0].hist([a0,b0], bins=32, stacked=True, color=['blue', 'red'])
    axs[1].hist([a2,b2], bins=32, stacked=True, color=['blue', 'red'])
    axs[0].set_title("Search Non Saturday Night")
    axs[1].set_title("Search Saturday Night")

    df_class = pd.concat([df_class0, df_class1])
    df['anomaly5'] = df_class['anomaly']
    # df['anomaly5'] = np.array(df['anomaly22'] == -1).astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = df.sort_values('date_time')
    df['date_time_int'] = pd.to_datetime(df['date_time']).astype('int64')
    a = df.loc[df['anomaly5'] == -1, ('date_time_int', 'price_usd')] #anomaly
    ax.plot(df['date_time_int'], df['price_usd'], color='blue', label='Normal')
    ax.scatter(a['date_time_int'],a['price_usd'], color='red', label='Anomaly')
    plt.legend()

    a = df.loc[df['anomaly5'] == 1, 'price_usd']
    b = df.loc[df['anomaly5'] == -1, 'price_usd']

    fig, axs = plt.subplots(figsize=(10, 6))
    axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])
    plt.show();

expedia = load_data()
df = filter_dataset(expedia)
view_anomalies(df)