# time_series_anomaly_detection_examples

## Table of Contents  

* [Introduction](#introduction)<a name="introduction"/>
* [Python](#python)<a name="python"/>

### Introduction

The Time Series Anomaly Detection repo contains several examples of anomaly detection algorithms for use with time series data sets.

Examples can be found in the [python](python) directory.

InfluxDB and Grafana are optionally included in the Docker stack for data storage and visualization purposes. CSV files can be easily imported to your InfluxDB instance using the [csv-to-influxdb](https://github.com/fabio-miranda/csv-to-influxdb) package. Telegraf has been included to fill the InfluxDB with dummy metric data.

Redis is optionally included in the Docker stack for caching (i.e. memoization) purposes.

### Python

Examples are typically written in python. From the [.env.example file](https://github.com/JBris/time_series_anomaly_detection_examples/blob/master/.env.example), you can see that scripts are written in python 3.8.2. A list of module dependencies can be found in the [Dockerfile](python/Dockerfile). You aren't particularly forced to use Docker, and can use something like Conda instead if that's your preference.

If you opt to use Docker, you can view the [Makefile](Makefile) for relevant Docker commands. The `make penter` command will create a new container and execute the python CLI. The `make prun` command will run a python script. For example, `make prun d=ts_price_anomaly_detection s=view` will run [ts_price_anomaly_detection/view.py](python/ts_price_anomaly_detection/view.py)

Example anomaly detection algorithms can be found in the [python](python) directory, and each example directory contains a similar structure. When exploring an example, you should first read the README.md and references.md files. The references.md file will provide you with a relevant link to a tutorial page and data set. Download the recommended data set and place it in the local data directory (don't place it in the [root data directory](data)).

You can then execute various python scripts to analyze and model the data. It's recommended that you run explore.py then view.py first to better understand the distribution of the data.
