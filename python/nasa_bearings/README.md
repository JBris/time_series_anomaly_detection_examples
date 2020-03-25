## Usage

Place data set in the local data directory. View the [references.md](references.md) file for more information.

## Importing to InfluxDB

```
csv-to-influxdb.py -i $(pwd)/data/merged_dataset_BearingTest_2.csv -s ${INFLUXDB_HOST}:8086 -tc timestamp \ 
--metricname value --fieldcolumns value --dbname ec2 --create \ 
-b 5000
```
