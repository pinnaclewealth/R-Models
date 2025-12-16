rm(list=ls())
set.seed(123)

# load crime and new city data
crime <- read.table("uscrime.txt", header = TRUE)

# load packages
if(!require(tree)) install.packages("tree")
library(tree)
if(!require(caret)) install.packages("caret")
library(caret)
if(!require(randomForest)) install.packages("randomForest")
library(randomForest)

# fit regression tree
model_tree <- tree(Crime ~., data = crime)
summary(model_tree)

model_tree$frame
plot(model_tree)
text(model_tree)

# determine best size for tree pruning
cv_tree <- cv.tree(model_tree, FUN = prune.tree)
cv_tree
plot(cv_tree$size, cv_tree$dev, type = 'b',
     xlab = "Tree size (#terminal nodes)", ylab = 'Deviance',
     main = 'Cross-Validation for Tree Pruning')

best_size <- cv_tree$size[which.min(cv_tree$dev)]
best_size

# prune tree to 5, 6, 7
tree5 <- prune.tree(model_tree, best = 5)
tree6 <- prune.tree(model_tree, best = 6)
tree7 <- prune.tree(model_tree, best = 7)

# visual of pruned tree
par(mfrow=c(1,3))
plot(tree5); text (tree5, pretty = 0, cex = 0.7); title("5-node tree")
plot(tree6); text (tree6, pretty = 0, cex = 0.7); title("6-node tree")
plot(tree7); text (tree7, pretty = 0, cex = 0.7); title("7-node tree")

#prune tree to 5
pruned_tree5 <- tree5
pruned_tree5$frame

# MSE for 5-node tree
pred_5node <- predict(pruned_tree5, newdata = crime)
mse_5node <- mean((pred_5node - crime$Crime)^2)
mse_5node


# part b Random Forest
rf_model <- randomForest(Crime ~., data = crime, importance = TRUE)
print(rf_model)
plot(rf_model)

# Variable Importance
importance(rf_model)
varImpPlot(rf_model)

# RF Model Performance
rf_pred <- predict(rf_model, newdata = crime)
mse_rf <- mean((rf_pred - crime$Crime)^2)
mse_rf


# 10.3
german <- read.table("germancredit.txt", header = FALSE)
head(german)
str(german)
german$V21 <- ifelse(german$V21 == 1, 1, 0)

# Fit logistic regression
logit_model <- glm(V21 ~., data = german, family = binomial(link = "logit"))
summary(logit_model)

# predicted probabilities
pred_probs <- predict(logit_model, type = "response")

# confusion matrix
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
table(Predicted = pred_class, Actual = german$V21)

# overall accuracy
mean(pred_class == german$V21)

# Adding cost threshold for bad customers
threshold <- 5/ (5+1)
pred_class_cost <- ifelse(pred_probs > threshold, 1, 0)

# new confusion matrix
table(Predicted = pred_class_cost, Actual = german$V21)
mean(pred_class_cost == german$V21)


