## Usage

Place data set in the local data directory. View the [references.md](references.md) file for more information.

## Importing to InfluxDB

```
csv-to-influxdb.py -i $(pwd)/data/spx.csv -s ${INFLUXDB_HOST}:8086 -tc date -tf '%Y-%m-%d' --metricname value --fieldcolumns close --dbname spx_500 --create -b 5000
```
