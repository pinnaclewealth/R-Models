rm(list=ls())
set.seed(123)

# load crime and new city data
crime <- read.table("uscrime.txt", header = TRUE)

y <- crime$Crime
x <- crime[, setdiff(names(crime), "Crime")]

# predictors as matrix
x_matrix <- as.matrix(x)

# regular linear model
crime_model <- lm(Crime~., data = crime)
summary(crime_model)

# Stepwise model (final model is minimizes AIC)
step_model <- step(crime_model, direction = 'both')
summary(step_model)

if(!require(glmnet)) install.packages("glmnet")
library(glmnet)

# LAsso Regression
x_scaled <- scale(x_matrix)
y_scaled <- scale(y)

# fit lasso
lasso_model <- glmnet(x_scaled, y_scaled, alpha = 1)

#cross validation for lambda (alpha)
cv_lasso <- cv.glmnet(x_scaled, y_scaled, alpha = 1)

# best lambda (alpha)
best_lambda_lasso <- cv_lasso$lambda.min

coef(cv_lasso, s = 'lambda.min')


# Elastic Net Regression
cv_enet <- cv.glmnet(x_scaled, y_scaled, alpha = 0.5)
best_lambda_enet <- cv_enet$lambda.min
coef(cv_enet, s = 'lambda.min')


for (a in c(0,0.25, 0.5, 0.75, 1)) {
  cv_model <- cv.glmnet(x_scaled, y_scaled, alpha = a)
  cat("Alpha:", a, " | Best Lambda:", cv_model$lambda.min, "\n")
}


plot(cv_lasso)
title("Lasso Cross Validation Curve")


cv_lasso$cvm[cv_lasso$lambda == cv_lasso$lambda.min]
cv_enet$cvm[cv_enet$lambda == cv_enet$lambda.min]
