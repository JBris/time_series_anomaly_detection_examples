## Usage

Place data set in the local data directory. View the [references.md](references.md) file for more information.

## Importing to InfluxDB

```
csv-to-influxdb.py -i $(pwd)/data/nyc_taxi.csv -s ${INFLUXDB_HOST}:8086 -tc timestamp --metricname value --fieldcolumns value --dbname taxi_demand --create -b 5000
```
