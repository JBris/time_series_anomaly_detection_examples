## Importing to Grafana

```
csv-to-influxdb.py -i $(pwd)/data/train.csv -s ${INFLUXDB_HOST}:8086 -tc date_time \ 
--metricname price_usd --fieldcolumns price_usd --dbname hotels --create \ 
-b 10000 --tagcolumns srch_saturday_night_bool, srch_booking_window
```
