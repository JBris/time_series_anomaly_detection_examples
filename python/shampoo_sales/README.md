## Usage

Place data set in the local data directory. View the [references.md](references.md) file for more information.

## Importing to InfluxDB

```
csv-to-influxdb.py -i $(pwd)/data/shampoo.csv -s ${INFLUXDB_HOST}:8086 -tc Month -tf '%Y-%m-%d' --metricname value --fieldcolumns Sales --dbname shampoo_sales --create -b 5000
```
