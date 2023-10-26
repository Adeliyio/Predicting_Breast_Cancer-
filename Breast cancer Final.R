# Load relevant libraries
library(randomForest)
library(dplyr)
library(ggplot2)
library(caTools)
library(class)
library(neuralnet)
library(MASS)
library(RColorBrewer)
library(gridExtra)

# Import Data
pd <- read.csv("data 2.csv")

# Removing columns "id" and "X"
pd <- pd[, !(names(pd) %in% c("id", "X"))]

# Check data structure
str(pd)
head(pd)
summary(pd)




# EDA of a few variables
plot1 <- ggplot(pd, aes(perimeter_mean, area_mean)) +
  geom_point(aes(color = factor(diagnosis)), alpha = 0.5) +
  scale_color_manual(name = "Diagnosis", values = brewer.pal(2, "Set1"), labels = c("M", "B")) +
  labs(title = "Diagnosis based on perimeter and area mean")

plot2 <- ggplot(pd, aes(symmetry_mean, smoothness_se)) +
  geom_point(aes(color = factor(diagnosis)), alpha = 0.5) +
  scale_color_manual(name = "Diagnosis", values = brewer.pal(2, "Set1"), labels = c("M", "B")) +
  labs(title = "Diagnosis based on symmetry and smoothness")

# Arrange plots side by side
grid.arrange(plot1, plot2, ncol = 2)



# Scale the data
scaled_pd <- as.data.frame(scale(pd[,-1]))
scaled_pd$diagnosis <- as.factor(pd$diagnosis)



# Convert diagnosis variable to binary numeric
pd$diagnosis <- ifelse(pd$diagnosis == "M", 1, 0)

# Remove highly correlated variables from the scaled dataset
numeric_scaled_pd <- scaled_pd[, sapply(scaled_pd, is.numeric)]
highly_correlated <- which(upper.tri(cor(numeric_scaled_pd), diag = TRUE) > 0.9)

# Logistic Regression
split <- sample.split(scaled_pd$diagnosis, SplitRatio = 0.7)
train <- subset(scaled_pd, split == TRUE)
test <- subset(scaled_pd, split == FALSE)

log_model <- glm(formula = diagnosis ~ ., family = binomial(link = 'logit'), data = train)
summary(log_model)
fitted_probabilities <- predict(log_model, newdata = test, type = 'response')

logreg_table <- table(test$diagnosis, fitted_probabilities > 0.5)
logreg_table
logreg_acc <- sum(diag(logreg_table)) / sum(logreg_table)
print(logreg_acc)

# Random Forest
rf_model <- randomForest(diagnosis ~ ., data = train)

# Remove the "diagnosis" column from the test data before making predictions
predicted_values <- predict(rf_model, test[,-which(names(test) == "diagnosis")])

rf_table <- table(predicted_values, test$diagnosis)
rf_table
rf_acc <- sum(diag(rf_table)) / sum(rf_table)
print(rf_acc)

# K Nearest Neighbors
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

pd1 <- as.data.frame(lapply(pd[2:30], normalize))
pd1$diagnosis <- pd$diagnosis

split <- sample.split(pd1$diagnosis, Split = 0.7)
train <- subset(pd1, split == T)
test <- subset(pd1, split == F)

predicted_values <- knn(train[,-1], test[,-1], train$diagnosis, k = 1)

mean(test$diagnosis != predicted_values)

predicted_values <- NULL
error_rate <- NULL

for (i in 1:10) {
  predicted_values <- knn(train[,-1], test[,-1], train$diagnosis, k = i)
  error_rate[i] <- mean(test$diagnosis != predicted_values)
}

k_values <- 1:10
error_df <- data.frame(error_rate, k_values)

pl <- ggplot(error_df, aes(x = k_values, y = error_rate)) + geom_point()
pl + geom_line(lty = "dotted", color = 'red')

predicted_values <- knn(train[,-1], test[,-1], train$diagnosis, k = 5)
mean(test$diagnosis != predicted_values)

knn_table <- table(test$diagnosis, predicted_values)
knn_table
knn_acc <- sum(diag(knn_table)) / sum(knn_table)
print(knn_acc)

# Neural Networks
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

pd1 <- as.data.frame(lapply(pd[2:31], normalize))
pd1$diagnosis <- as.numeric(pd$diagnosis)

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

pd1$diagnosis <- sapply(pd1$diagnosis, binary)

split <- sample.split(pd1$diagnosis, Split = 0.7)
train <- subset(pd1, split == T)
test <- subset(pd1, split == F)

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
nn

predicted_nn_values <- compute(nn, test[, 1:30])

predictions <- sapply(predicted_nn_values$net.result, round)

nn_table <- table(predictions, test$diagnosis)
nn_table
nn_acc <- sum(diag(nn_table)) / sum(nn_table)
print(nn_acc)

# Linear Discriminant Analysis (LDA)
lda.model <- lda(diagnosis ~ ., data = scaled_pd)  # Use the full dataset to build the LDA model
lda.predictions <- predict(lda.model, newdata = test)
lda_table <- table(test$diagnosis, lda.predictions$class)
lda_table
lda_acc <- sum(diag(lda_table)) / sum(lda_table)
print(lda_acc)

# Quadratic Discriminant Analysis (QDA)
qda.model <- qda(diagnosis ~ ., data = scaled_pd)  # Use the full dataset to build the QDA model
qda.predictions <- predict(qda.model, newdata = test)
qda_table <- table(test$diagnosis, qda.predictions$class)
qda_table
qda_acc <- sum(diag(qda_table)) / sum(qda_table)
print(qda_acc)

# Accuracy
accuracies <- matrix(c(logreg_acc, rf_acc, knn_acc, nn_acc, lda_acc, qda_acc), ncol = 1, byrow = FALSE)
colnames(accuracies) <- c("Accuracy")
rownames(accuracies) <- c("LogReg", "RandomForest", "KNN", "NeuralNetwork", "LDA", "QDA")
accuracies <- as.table(accuracies)
accuracies
