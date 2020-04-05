# time_series_anomaly_detection_examples

## Table of Contents  

* [Introduction](#introduction)<a name="introduction"/>
* [Python](#python)<a name="python"/>
* [R](#r)<a name="r"/>
* [InfluxDB](#influxdb)<a name="influxdb"/>

### Introduction

The Time Series Anomaly Detection repo contains several examples of anomaly detection algorithms for use with time series data sets.

Examples can be found in the [python directory](python) and [r directory](r).

InfluxDB and Grafana are optionally included in the Docker stack for data storage and visualization purposes. Telegraf has been included to fill the InfluxDB with dummy metric data.

Redis is optionally included in the Docker stack for caching (i.e. memoization) purposes.

If you're using Docker, execute [build.sh](build.sh) to get started.

### Python

Examples are typically written in python. From the [.env.example file](.env.example), you can see that scripts are written in python 3.8.2. A list of module dependencies can be found in the [Dockerfile](python/Dockerfile) and [requirements.txt](python/requirements.txt). You aren't forced to use Docker, and can use something like Conda instead if that's your preference.

If you opt to use Docker, you can view the [Makefile](Makefile) for relevant Docker commands. The `make penter` command will create a new container and execute the python CLI. The `make prun` command will run a python script. For example, `make prun d=ts_price_anomaly_detection s=view` will run [ts_price_anomaly_detection/view.py](python/ts_price_anomaly_detection/view.py)

Example anomaly detection algorithms can be found in the [python](python) directory, and each example directory contains a similar structure. When exploring an example, you should first read the README.md and references.md files. The references.md file will provide you with a relevant link to a tutorial page and data set. Download the recommended data set and place it in the local data directory (don't place it in the [root data directory](data)).

You can then execute various python scripts to analyze and model the data. It's recommended that you run explore.py then view.py first to better understand the distribution of the data.

### R

Additional examples are written in R. From the [.env.example file](.env.example), you can see that R scripts are written in version 3.6.3. A list of additional R packages can be found in the [Dockerfile](r/Dockerfile). 

As the [docker-compose.yml](docker-compose.yml) file shows, this repo employs the [rocker/tidyverse image](https://hub.docker.com/r/rocker/tidyverse) which already includes the tidyverse collection and RStudio server.

If you opt to use Docker, you can view the [Makefile](Makefile) for relevant Docker commands. The `make renter` command will allow users to execute shell commands within the R container. The `make prun` command will run an R script. For example, `make rrun s=bitcoin_anomalies` will run [$R_STUDIO_USER/view.r](r/user/bitcoin_anomalies.r)

Example anomaly detection algorithms can be found in the [r](r) directory. You can then execute various r scripts to analyze and model the data. 

### InfluxDB

InfluxDB is a time series database. For those who are unfamiliar, more information can be found at [influxdata.com](https://www.influxdata.com/). InfluxDB can be combined with [Grafana](https://grafana.com/) to analyze and visualize the data. View the [.env.example file](.env.example) to configure your InfluxDB & Grafana versions and ports.

CSV files can be easily imported to your InfluxDB instance using the [csv-to-influxdb](https://github.com/fabio-miranda/csv-to-influxdb) package. Each example directory will contain a README.md file with a `csv-to-influxdb.py` command to execute.
