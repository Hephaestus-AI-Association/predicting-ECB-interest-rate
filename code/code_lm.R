data <- read.csv(file.choose(), header=T) # filled_data.csv
BCI <- read.csv(file.choose(), header=T) # BCI.csv
LT <- read.csv(file.choose(), header=T) # LT_intrates.csv
ST <- read.csv(file.choose(), header=T) # ST_intrates.csv

# Plot historical datas
library(ggplot2)
ggplot(data, aes(x=data$Date, y=data$Interest_Rate)) +
  geom_line()
ggplot(data, aes(x=data$Date, y=data$Inflation)) +
  geom_line()

# Extract data from Jan-1999 to Jan-2023
BCI <- BCI[3:291,]
LT <- LT[1:289,]
ST <- ST[1:289,]

# Add new columns to data dataset
data['BCI'] = BCI$Value
data['LT'] = LT$Value
data['ST'] = ST$Value
View(data)

# Correlation Matrix of covariates
cor_matrix <- cor(data[, c("Inflation", "Interest_Rate", "ST", "LT", "BCI")])
corrplot(cor_matrix,
         method = "color",
         tl.col = "black",
         tl.srt = 45,
         main = "Correlation Heatmap"
)

# First model
rm(BCI)
rm(LT)
rm(ST)
attach(data)
model <- lm(Interest_Rate ~ Inflation + BCI + LT + ST)
summary(model)
qqnorm(residuals(model))
qqline(residuals(model))
plot(model, which = 1)
plot(fitted(model), residuals(model))
plot(Interest_Rate, predict(model))

# Model removing BCI
model <- lm(Interest_Rate ~ Inflation + data$LT + data$ST)
summary(model)
qqnorm(residuals(model))
qqline(residuals(model))
plot(fitted(model), residuals(model))
plot(Interest_Rate, predict(model))
abline(0,1)
mse <- mean((predict(model)-Interest_Rate)^2)
mse

# Find outliers
boxplot(Inflation, outline=T, main="Inflation Rate Boxplot")
outliers_in <- boxplot.stats(Inflation)$out
outliers_in
boxplot(data$LT, outline=T)
outliers <- boxplot.stats(data$LT)$out
outliers_LT
boxplot(data$ST, outline=T)
outliers <- boxplot.stats(data$ST)$out
outliers_ST

# Model removing outliers
no_outliers_data <- subset(data, !(Inflation %in% outliers_in))
model <- lm(sqrt(Interest_Rate+3) ~ Inflation + LT + ST, data=no_outliers_data)
summary(model)
qqnorm(residuals(model))
qqline(residuals(model), col="red", lwd=2)
plot(fitted(model), residuals(model), main="Fitted vs Residuals")
plot(sqrt(no_outliers_data$Interest_Rate+3), predict(model), xlab="sqrt(Interest_Rate+3)", main="Predicted vs True values")
abline(0,1, lwd=2, col="red")
mse <- mean((predict(model)-no_outliers_data$Interest_Rate)^2)
mse
plot(model, which = 1)

