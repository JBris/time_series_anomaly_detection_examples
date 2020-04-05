library(anomalize) #tidy anomaly detectiom
library(tidyverse) #tidyverse packages like dplyr, ggplot, tidyr
library(coindeskr) #bitcoin price extraction from coindesk

btc <- get_historic_price(start = "2017-01-01")

#Convert to time series
btc_ts <- btc %>% rownames_to_column() %>% as_tibble() %>% 
  mutate(date = as.Date(rowname)) %>% select(-one_of('rowname'))
head(btc_ts)

#Decompose time series
btc_ts %>% 
  time_decompose(Price, method = "stl", frequency = "auto", trend = "auto") %>%
  anomalize(remainder, method = "gesd", alpha = 0.05, max_anoms = 0.2) %>%
  plot_anomaly_decomposition()

# Detect anomalies
btc_ts %>% 
  time_decompose(Price) %>%
  anomalize(remainder) %>%
  time_recompose() %>%
  plot_anomalies(time_recomposed = TRUE, ncol = 3, alpha_dots = 0.5)

# Extract anomaly obervations
btc_ts %>% 
  time_decompose(Price) %>%
  anomalize(remainder) %>%
  time_recompose() %>%
  filter(anomaly == 'Yes') 