#!/usr/bin/env python

import json
import os
import pandas as pd
import pickle
from cache import get, set

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
                                    "srch_room_count": "int16",
                                    "visitor_location_country_id": "int16",
                                    "srch_booking_window": "int16", 
                                    "srch_saturday_night_bool": "bool", 
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
