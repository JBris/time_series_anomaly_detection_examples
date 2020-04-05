# Source: https://www.business-science.io/code-tools/2018/04/08/introducing-anomalize.html
# Author: Matt Dancho - https://resources.rstudio.com/authors/matt-dancho

library(tidyverse)
library(anomalize)

tidyverse_cran_downloads

# Detect anomalies
tidyverse_cran_downloads %>%
  time_decompose(count) %>%
  anomalize(remainder) %>%
  time_recompose() %>%
  plot_anomalies(time_recomposed = TRUE, ncol = 3, alpha_dots = 0.5)

# Decomposition
tidyverse_cran_downloads %>%
  time_decompose(count, method = "stl", frequency = "auto", trend = "auto")

# Detect anomalies
tidyverse_cran_downloads %>%
  time_decompose(count, method = "stl", frequency = "auto", trend = "auto") %>%
  anomalize(remainder, method = "iqr", alpha = 0.05, max_anoms = 0.2)

# Visualize lubridate
tidyverse_cran_downloads %>%
  
# Select a single time series
filter(package == "lubridate") %>%
  ungroup() %>%
  
# Anomalize
time_decompose(count, method = "stl", frequency = "auto", trend = "auto") %>%
  anomalize(remainder, method = "iqr", alpha = 0.05, max_anoms = 0.2) %>%
  
# Plot Anomaly Decomposition
plot_anomaly_decomposition() +
ggtitle("Lubridate Downloads: Anomaly Decomposition")

# Recompose time series
tidyverse_cran_downloads %>%
  time_decompose(count, method = "stl", frequency = "auto", trend = "auto") %>%
  anomalize(remainder, method = "iqr", alpha = 0.05, max_anoms = 0.2) %>%
  time_recompose()

# Visualize Lubridate
tidyverse_cran_downloads %>%
  # Select single time series
  filter(package == "lubridate") %>%
  ungroup() %>%
  # Anomalize
  time_decompose(count, method = "stl", frequency = "auto", trend = "auto") %>%
  anomalize(remainder, method = "iqr", alpha = 0.05, max_anoms = 0.2) %>%
  time_recompose() %>%
  # Plot Anomaly Decomposition
  plot_anomalies(time_recomposed = TRUE) +
  ggtitle("Lubridate Downloads: Anomalies Detected")