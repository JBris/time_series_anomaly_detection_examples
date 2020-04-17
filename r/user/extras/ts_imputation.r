# Source: https://www.kaggle.com/juejuewang/handle-missing-values-in-time-series-for-beginners
# Author: Nishant Singh - https://www.datacamp.com/profile/NishantKumarSingh

library(imputeTS)

View(tsAirgap)

plot(tsAirgap, main="AirPassenger data with missing values")

statsNA(tsAirgap)

#########################################################################################

# General imputation methods

par(mfrow=c(2,2))
# Mean Imputation
plot(na_mean(tsAirgap, option = "mean") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Mean")
mean((na_mean(tsAirgap, option = "mean") - AirPassengers)^2)

# Median Imputation
plot(na_mean(tsAirgap, option = "median") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Median")
mean((na_mean(tsAirgap, option = "median") - AirPassengers)^2)

# Mode Imputation
plot(na_mean(tsAirgap, option = "mode") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Mode")
mean((na_mean(tsAirgap, option = "mode") - AirPassengers)^2)    

# Random Imputation
plot(na_random(tsAirgap) - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Random")
mean((na_random(tsAirgap) - AirPassengers)^2)

#########################################################################################

#TS specific imputation methods

par(mfrow=c(2,2))
# Last Observartion Carried Forward
plot(na_locf(tsAirgap, option = "locf") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "LOCF")
m1 <- mean((na_locf(tsAirgap, option = "locf") - AirPassengers)^2)

# Next Observartion Carried Backward
plot(na_locf(tsAirgap, option = "nocb") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "NOCB")
m2 <- mean((na_locf(tsAirgap, option = "nocb") - AirPassengers)^2)

# Linear Interpolation
plot(na_interpolation(tsAirgap, option = "linear") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Linear")
m3 <- mean((na_interpolation(tsAirgap, option = "linear") - AirPassengers)^2)

# Spline Interpolation
plot(na_interpolation(tsAirgap, option = "spline") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Spline")

m4 <- mean((na_interpolation(tsAirgap, option = "spline") - AirPassengers)^2)

data.frame(methods=c('LOCF', 'NACB', 'Linear', 'Spline'), MSE=c(m1, m2, m3, m4))

#########################################################################################

#Combined imputation approach

par(mfrow=c(2,2))
# Seasonal Adjustment then Random
plot(na_seadec(tsAirgap, algorithm = "random") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Seas-Adj -> Random")
ma1 <- mean((na_seadec(tsAirgap, algorithm = "random") - AirPassengers)^2)

# Seasonal Adjustment then Mean
plot(na_seadec(tsAirgap, algorithm = "mean") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Seas-Adj -> Mean")
ma2 <- mean((na_seadec(tsAirgap, algorithm = "mean") - AirPassengers)^2)

# Seasonal Adjustment then LOCF
plot(na_seadec(tsAirgap, algorithm = "locf") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Seas-Adj -> LOCF")
ma3 <- mean((na_seadec(tsAirgap, algorithm = "locf") - AirPassengers)^2)

# Seasonal Adjustment then Linear Interpolation
plot(na_seadec(tsAirgap, algorithm = "interpolation") - AirPassengers, ylim = c(-mean(AirPassengers), mean(AirPassengers)), ylab = "Difference", main = "Seas-Adj -> Linear")

ma4 <- mean((na_seadec(tsAirgap, algorithm = "interpolation") - AirPassengers)^2)

data.frame(methods=c("Seas-Adj+Random", "Seas-Adj+Mean", "Seas-Adj+LOCF","Seas-Adj+Linear"),
           MSE=c(ma1, ma2, ma3, ma4))
           