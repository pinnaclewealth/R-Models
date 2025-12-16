rm(list=ls())
set.seed(123)

# load crime and new city data
crime <- read.table("uscrime.txt", header = TRUE)

y <- crime$Crime
x <- crime[, setdiff(names(crime), "Crime")]

new_city <- data.frame(
  M = 14.0,
  So = 0,
  Ed = 10.0,
  Po1 = 12.0,
  Po2 = 15.5,
  LF = 0.640,
  M.F = 94.0,
  Pop = 150,
  NW = 1.1,
  U1 = 0.120,
  U2 = 3.6,
  Wealth = 3200,
  Ineq = 20.1,
  Prob = 0.04,
  Time = 39.0
)

# PCA with scaling
pca_model <- prcomp(x, scale.=TRUE)

# Variance explained
summary(pca_model)

# First 6 PCs selected to reach 90% of variance
pc_data <- data.frame(y, pca_model$x[,1:6])
pc_reg <- lm(y ~., data = pc_data)
summary(pc_reg)


# back-transform coefficients
loads <- pca_model$rotation[,1:6] #PCA loadings
betas_pc <- pc_reg$coefficients[-1] #Coeffs for PCs
scales <- apply(x, 2, sd) #scaling factors
means <- apply(x, 2, mean) # means used

# Back-transform PC to original
beta_orig <- loads %*% betas_pc / scales
beta0 <- pc_reg$coefficients[1] - sum(beta_orig*means)

final_model <- c(beta0, beta_orig)
names(final_model) <-c("Intercept", colnames(x))
final_model


# Prediction for new city
pred_pca <- final_model["Intercept"] + as.matrix(new_city) %*% final_model[-1]
pred_pca

# Compare with Q8.2 Regression
lm_orig  <- lm(Crime~., data = crime)
summary(lm_orig)

# Prediction from original regression
pred_orig <- predict(lm_orig, new_city)
pred_orig

# Prediction with significant factors for original regression
lm_orig2 <- lm(Crime ~  M + Ed + Po1 + U2 + Ineq + Prob, data = crime )
pred2_orig <- predict(lm_orig2, new_city)
pred2_orig

if(!require(DAAG)) install.packages("DAAG")
library(DAAG)

# 5 - fold cross validation
c1 <- cv.lm(crime,lm_orig2,m=5)

if(!require(caret)) install.packages("caret")
library(caret)
if(!require(pls)) install.packages("pls")
library(pls)


# Train PCA regression with CV
train_control <- trainControl(method = "cv", number = 5)
pca_cv <- train(x,y, method = "pcr", trControl = train_control,
                preProcess = c("center","scale"),
                tuneLength = 10)

# best number of PCs
best_ncomp <- pca_cv$bestTune$ncomp
cat("Optimal number of PCs (via CV): ", best_ncomp, "\n")

# results
print(pca_cv$results[pca_cv$results$ncomp == best_ncomp, ])

# Prediction for new city data with CV-optimal 
pred_cv <- predict(pca_cv, new_city)
cat("Prediction for new city (CV-optimized PCR): ", pred_cv, "\n")


# Compare results
cat("\n--- Model Comparison ---\n")
cat("PCA Regression Prediction (manual, 6 PCs): ", pred_pca, "\n")
cat("Original Regression Prediction (all variables): ", pred_orig, "\n")
cat("Reduced Regression Prediction (sig. variables): ", pred2_orig, "\n")
cat("CV-Optimized PCR Prediction: ", pred_cv, "\n\n")


cat("Adj R2 PCA Regression  (manual, 6 PCs): ", summary(pc_reg)$adj.r.squared, "\n")
cat("Adj R2 Original Regression (all variables): ", summary(lm_orig)$adj.r.squared, "\n")
cat("Adj R2 Reduced Regression (sig. variables): ", summary(lm_orig2)$adj.r.squared, "\n")
cat("Best PCR CV R2: ", max(pca_cv$results$Rsquared), "\n")

if(!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

results_r <- data.frame(
  Model = c("PCA (6 PCs)", "Original (all vars)", "Reduced (sig vars)", "PCR (CV-optimized)"),
  R2 = c(summary(pc_reg)$adj.r.squared,
         summary(lm_orig)$adj.r.squared,
         summary(lm_orig2)$adj.r.squared,
         max(pca_cv$results$Rsquared)
         )
)

ggplot(results_r, aes(x = Model, y = R2, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(R2, 3)), vjust = -0.5, size = 4) +
  labs(
    title = "Model Performance Comparison",
    y = "R-Squared",
    x = ""
  ) +
  theme_minimal() +
  theme(legend.position = "none")

results_p <- data.frame(
  Model = c("PCA (6 PCs)", "Original (all vars)", "Reduced (sig vars)", "PCR (CV-optimized)"),
  Prediction = c(
    as.numeric(pred_pca),
    as.numeric(pred_orig),
    as.numeric(pred2_orig),
    as.numeric(pred_cv)
  )
)

ggplot(results_p, aes(x = Model, y = Prediction, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(Prediction, 1)), vjust = -0.5, size = 4) +
  labs(
    title = "Model Performance Comparison",
    y = "Predicted Crime",
    x = ""
  ) +
  theme_minimal() +
  theme(legend.position = "none")
