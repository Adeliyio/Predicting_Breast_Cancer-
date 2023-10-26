

library(randomForest)
library(Amelia)
library(dplyr)
library(ggplot2)
library(caTools)
library(stringr)
library(randomForest)
library(caTools)
library(rpart.plot)
library(corrplot)
library(Hmisc)
library(rpart.plot)
library(class)
library(e1071)
library(neuralnet)
library(naniar)
library(tidyverse)
library(MASS)
library(dplyr)


# Import Data and check structure
boobs <- read.csv("data 2.csv")
names(boobs)

# Removing columns "id" and "X"
boobs <- boobs[, !(names(boobs) %in% c("id", "X"))]

# Scale the data
scaled_boobs <- as.data.frame(scale(boobs[,-1]))
scaled_boobs$diagnosis <- as.factor(boobs$diagnosis)

# Check the structure of the scaled dataset
str(scaled_boobs)
head(scaled_boobs)

# Columns highly correlated with diagnosis and data visualization
ggplot(boobs, aes(perimeter_mean, area_mean)) +
  geom_point(aes(color = factor(diagnosis)), alpha = 0.5) +
  scale_fill_discrete(name = "diagnosis", breaks = c("0", "1"), labels = c("M", "B")) +
  labs(title = "Diagnosis based on perimeter and area mean")

ggplot(boobs, aes(symmetry_mean, smoothness_se)) +
  geom_point(aes(color = factor(diagnosis)), alpha = 0.5) +
  scale_fill_discrete(name = "diagnosis", breaks = c("0", "1"), labels = c("M", "B")) +
  labs(title = "Diagnosis based on symmetry and smoothness")

# Convert diagnosis variable to binary numeric
boobs$diagnosis <- ifelse(boobs$diagnosis == "M", 1, 0)

# Remove highly correlated variables from the scaled dataset
numeric_scaled_boobs <- scaled_boobs[, sapply(scaled_boobs, is.numeric)]
highly_correlated <- which(upper.tri(cor(numeric_scaled_boobs), diag = TRUE) > 0.9)


# Scale the data
scaled_boobs <- as.data.frame(scale(boobs[,-1]))  # -1 to exclude the target variable (assuming it's in the first column)

# Add back the diagnosis column
scaled_boobs$diagnosis <- as.factor(boobs$diagnosis)

# Logistic Regression
split <- sample.split(scaled_boobs$diagnosis, SplitRatio = 0.7)
train <- subset(scaled_boobs, split == TRUE)
test <- subset(scaled_boobs, split == FALSE)

log.model <- glm(formula = diagnosis ~ ., family = binomial(link = 'logit'), data = train)
summary(log.model)
fitted.probabilities <- predict(log.model, newdata = test, type = 'response')

c <- table(test$diagnosis, fitted.probabilities > 0.5)
c.t <- sum(diag(c)) / sum(c)
print(c.t)


# Random Forest
rf.model <- randomForest(diagnosis ~ ., data = train)

# Remove the "diagnosis" column from the test data before making predictions
predicted.values <- predict(rf.model, test[,-which(names(test) == "diagnosis")])

d <- table(predicted.values, test$diagnosis)
d.t <- sum(diag(d)) / sum(d)
print(d.t)

# K Nearest Neighbors
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

boobs1 <- as.data.frame(lapply(boobs[2:30], normalize))
boobs1$diagnosis <- boobs$diagnosis

split <- sample.split(boobs1$diagnosis, Split = 0.7)
train <- subset(boobs1, split == T)
test <- subset(boobs1, split == F)

predicted.boobs <- knn(train[,-1], test[,-1], train$diagnosis, k = 1)

mean(test$diagnosis != predicted.boobs)

predicted.boobs <- NULL
error.rate <- NULL

for (i in 1:10) {
  predicted.boobs <- knn(train[,-1], test[,-1], train$diagnosis, k = i)
  error.rate[i] <- mean(test$diagnosis != predicted.boobs)
}

k.values <- 1:10
error.df <- data.frame(error.rate, k.values)

pl <- ggplot(error.df, aes(x = k.values, y = error.rate)) + geom_point()
pl + geom_line(lty = "dotted", color = 'red')

predicted.boobs <- knn(train[,-1], test[,-1], train$diagnosis, k = 5)
mean(test$diagnosis != predicted.boobs)

e <- table(test$diagnosis, predicted.boobs)
print(e)
e.t <- sum(diag(e)) / sum(e)
print(e.t)

# Neural Networks
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

boobs1 <- as.data.frame(lapply(boobs[2:31], normalize))
boobs1$diagnosis <- as.numeric(boobs$diagnosis)

binary <- function(dg) {
  for (i in 1:length(dg)) {
    if (dg[i] == 1) {
      dg[i] <- 0
    } else {
      dg[i] <- 1
    }
  }
  return(dg)
}

boobs1$diagnosis <- sapply(boobs1$diagnosis, binary)

split <- sample.split(boobs1$diagnosis, Split = 0.7)
train <- subset(boobs1, split == T)
test <- subset(boobs1, split == F)

nn <- neuralnet(
  diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + 
    smoothness_mean + compactness_mean + concavity_mean + concave.points_mean + 
    symmetry_mean + fractal_dimension_mean + radius_se + texture_se + 
    perimeter_se + area_se + smoothness_se + compactness_se + 
    concavity_se + concave.points_se + symmetry_se + fractal_dimension_se + 
    radius_worst + texture_worst + perimeter_worst + area_worst + 
    smoothness_worst + compactness_worst + concavity_worst + 
    concave.points_worst + symmetry_worst + fractal_dimension_worst, 
  data = train, hidden = c(5, 3), linear.output = FALSE
)

predicted.nn.values <- compute(nn, test[, 1:30])

predictions <- sapply(predicted.nn.values$net.result, round)

g <- table(predictions, test$diagnosis)
g.t <- sum(diag(g)) / sum(g)
print(g.t)

library(MASS)
lda.model <- lda(diagnosis ~ ., data = scaled_boobs)  # Use the full dataset to build the LDA model
lda.predictions <- predict(lda.model, newdata = test)
lda.table <- table(test$diagnosis, lda.predictions$class)
lda.t <- sum(diag(lda.table)) / sum(lda.table)

# Quadratic Discriminant Analysis (QDA)
qda.model <- qda(diagnosis ~ ., data = scaled_boobs)  # Use the full dataset to build the QDA model
qda.predictions <- predict(qda.model, newdata = test)
qda.table <- table(test$diagnosis, qda.predictions$class)
qda.t <- sum(diag(qda.table)) / sum(qda.table)

# Accuracy
accur <- matrix(c(c.t, d.t, e.t, g.t, lda.t, qda.t), ncol = 1, byrow = FALSE)
colnames(accur) <- c("Accuracy")
rownames(accur) <- c("LG", "RF", "KNN", "NN", "LDA", "QDA")
accur <- as.table(accur)
accur

