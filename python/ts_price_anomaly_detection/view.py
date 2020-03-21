#!/usr/bin/env python

import os
import pandas as pd
from cache import get, set

dirname = os.path.dirname(os.path.realpath(__file__))

expedia = get('expedia_df')
if expedia is None:
    filename = dirname + '/data/train.csv'
    chunksize = 10 ** 5
    TextFileReader = pd.read_csv(filename, chunksize=chunksize)
    expedia = pd.concat(TextFileReader, ignore_index=True)
    set('expedia_df', expedia.to_pickle())
else:
    expedia = expedia.read_pickle()

# df = expedia.loc[expedia['prop_id'] == 104517]
# df = df.loc[df['srch_room_count'] == 1]
# df = df.loc[df['visitor_location_country_id'] == 219]
# df = df[['date_time', 'price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
# df.info()
# print(df.info())