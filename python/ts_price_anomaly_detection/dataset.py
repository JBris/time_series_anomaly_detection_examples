#!/usr/bin/env python

import json
import os
import pandas as pd
import pickle
from cache import get, set
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    expedia_str = get('expedia_df')
    if expedia_str is None:
        filename = dirname + '/data/train.csv'
        df_chunk = pd.read_csv(
                                filename, 
                                chunksize=10**6, 
                                usecols=['prop_id', 'srch_room_count', 'visitor_location_country_id', 'srch_booking_window', 'srch_saturday_night_bool', 'price_usd', 'date_time'], 
                                dtype={
                                    "prop_id": "int32",
                                    "srch_room_count": "int32",
                                    "visitor_location_country_id": "int32",
                                    "srch_booking_window": "int32", 
                                    "srch_saturday_night_bool": "int32", 
                                    "price_usd":"float32", 
                                    "date_time": "object"
                                }
                            )
        chunks = []
        for chunk in df_chunk: 
            chunks.append(chunk)
        expedia = pd.concat(chunks)
        set('expedia_df', pickle.dumps(expedia))
    else:
        expedia = pickle.loads(expedia_str)
    
    return expedia

def filter_dataset(expedia):
    df = expedia.loc[expedia['prop_id'] == 104517]
    df = df.loc[df['srch_room_count'] == 1]
    df = df.loc[df['visitor_location_country_id'] == 219]
    df = df.loc[df['price_usd'] < 5584]
    return df

def reindex_data(df):
    data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
    X = data.values
    X_std = StandardScaler().fit_transform(X)
    data = pd.DataFrame(X_std)
    # reduce to 2 important features
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    # standardize these 2 new features
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    return pd.DataFrame(np_scaled)