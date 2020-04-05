# Source: https://www.datacamp.com/community/tutorials/detect-anomalies-anomalize-r
# Author: Nishant Singh - https://www.datacamp.com/profile/NishantKumarSingh
# See https://cran.r-project.org/web/packages/purrr/index.html

library(tidyverse)
library(anomalize)

# Using data package provided in the anomalize package and taking single time series of package purrr
purrr_package = tidyverse_cran_downloads%>%
  filter(package == "purrr")%>%
  ungroup()

# Decompose
purrr_anomaly  = purrr_package %>%
  time_decompose(count)

purrr_anomaly%>% glimpse()

# Lower and upper bound transformation
purrr_anomaly = purrr_anomaly%>%
  time_recompose()

purrr_anomaly%>% glimpse()

# Plot anomalies
purrr_anomaly%>%
  plot_anomaly_decomposition()+
  ggtitle("Plotting Anomalies")

# Adjusting parameters of decomposition
purrr_package %>%
  time_decompose(count, frequency = "auto", trend = "2 weeks")%>%
  anomalize(remainder)%>%
  plot_anomaly_decomposition()+
  ggtitle("Trend = 2 Weeks / Frequency = auto ")

purrr_package%>%
  time_decompose(count)%>%
  anomalize(remainder, alpha = 0.025)%>%
  time_recompose()%>%
  plot_anomalies(time_recompose = T)+
  ggtitle("alpha = 0.025")

# Set max_anoms
purrr_package%>%
  time_decompose(count)%>%
  anomalize(remainder, alpha = 0.2, max_anoms = 0.2)%>%
  time_recompose()%>%
  plot_anomalies(time_recompose = T)+
  ggtitle("20% anomaly Allowed")

purrr_package%>%
  time_decompose(count)%>%
  anomalize(remainder, alpha = 0.2, max_anoms = 0.05)%>%
  time_recompose()%>%
  plot_anomalies(time_recompose = T)+
  ggtitle("5% anomaly Allowed")