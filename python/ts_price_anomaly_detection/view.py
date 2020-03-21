#!/usr/bin/env python

import os
import pandas as pd
from cache import get, set

dirname = os.path.dirname(os.path.realpath(__file__))
expedia = get('expedia_df')
if expedia is None:
    filename = dirname + '/data/train.csv'
    df_chunk = pd.read_csv(
                            filename, 
                            chunksize=10**6, 
                            usecols=['srch_booking_window', 'srch_saturday_night_bool', 'price_usd', 'date_time'], 
                            dtype={"srch_booking_window": "int32", "srch_saturday_night_bool": "bool", "price_usd":"float32", "date_time": "object"}
                        )
    chunks = []
    for chunk in df_chunk: 
        chunks.append(chunk)
    expedia = pd.concat(chunks)
    set('expedia_df', expedia.to_pickle())
else:
    expedia = expedia.read_pickle()

# df = expedia.loc[expedia['prop_id'] == 104517]
# df = df.loc[df['srch_room_count'] == 1]
# df = df.loc[df['visitor_location_country_id'] == 219]
# df = df[['date_time', 'price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
# df.info()
# print(df.info())

