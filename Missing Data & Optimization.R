rm(list=ls())


library(tidyverse)
library(recipes)
library(mice)
library(caret)
library(kernlab)
library(pROC)


set.seed(123)

datafile <- "breast-cancer-wisconsin.data.txt"

col_names <- c("id", "clump_thickness", "uniform_cell_size", "uniform_cell_shape", "marginal_adhesion", "single_epithelial_cell_size", 
               "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class")

df <- read.csv(datafile, header = FALSE, na.strings = c("?", "NA", ""), col.names = col_names)

head(df)
tail(df)
summary(df)

# drop ID column
df <- df %>% select(-id)
head(df)

df$class <- as.integer(as.character(df$cla))
# Map class {2,4} to {Benign, Malignant}
df <- df %>% 
  mutate(
    class = factor(dplyr::recode(
      class, '2' =  "Benign", '4' = "Malignant",
    ), levels = c("Benign", "Malignant"))
  )

# predictors to numeric
predictor_cols <- setdiff(names(df), "class")
df[predictor_cols] <- lapply(df[predictor_cols], function(x) as.numeric(as.character(x)))

# check columns with missing data
message("Rows: ", nrow(df))
message("Missing by column: ")
print(summarise_all(df, ~ sum(is.na(.))))

# 4 Model Evaluation (f-fold CV, caret)

ctrl <- trainControl(
  method = 'cv',
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary, 
  savePredictions = 'final',
  allowParallel = TRUE
)

# evaluate the methods
fit_score <- function(df, label = "dataset") {
  df <- df %>% 
    dplyr::filter(!is.na(class)) %>%
    dplyr::mutate(class = factor(class, levels = c("Benign", "Malignant")))
  x_cols <- setdiff(names(df), "class")
  preproc <- c("center", "scale")
  
  #SVM (radial)
  svm_fit <- caret::train( x = df[, x_cols], y = df$class, method = "svmRadial", metric = "ROC", trControl = ctrl, preProcess = preproc, tuneLength = 10)
  
  # KNN
  knn_fit <- caret::train( x = df[, x_cols], y = df$class, method = "knn", metric = "ROC", trControl = ctrl, preProcess = preproc, tuneLength = 10)
  
  # summarize the results each
  summarize_metrics <- function(fit, algo) {
    preds <- fit$pred
    
    # keep best tuning rows
    preds <- dplyr::inner_join(preds, fit$bestTune, by = names(fit$bestTune))
    if("Malignant" %in% names(preds)) {
      preds <- preds %>% dplyr::filter(!is.na(Malignant), !is.na(obs), !is.na(pred))
    }
    # drop folds with only one class (AUC/F1 undefined)
    preds <- preds %>%
      dplyr::group_by(Resample) %>%
      dplyr::filter(dplyr::n_distinct(obs) == 2) %>%
      dplyr::ungroup()
    
    if (nrow(preds) < 2) {
      # fallback: use aggregate resampled metrics at bestTune
      res <- dplyr::inner_join(fit$results, fit$bestTune, by = names(fit$bestTune)) %>% dplyr::slice(1)
      return(tibble::tibble(
        dataset   = label,
        algorithm = algo,
        Accuracy  = as.numeric(res$Accuracy),
        F1        = NA_real_,
        ROC_AUC   = as.numeric(res$ROC)
      ))
    }
    
    acc <- mean(preds$pred == preds$obs)
    f1 <- tryCatch(
      caret::F_meas(preds$pred, preds$obs, relevant = "Malignant"),
      error = function(e) NA_real_
    )
    
    auc <- tryCatch(
      pROC::roc(response = preds$obs,
                     predictor = preds$Malignant,
                     levels = c("Benign", "Malignant"),
                     quiet = TRUE)$auc,
      error = function(e) NA_real_
    )
    
    tibble(dataset = label, algorithm = algo, 
           Accuracy = as.numeric(acc),
           F1 = as.numeric(f1),
           ROC_AUC = as.numeric(auc))
  }
  
  bind_rows(
    summarize_metrics(svm_fit, "SVM (radial)"), 
    summarize_metrics(knn_fit, "KNN")
  )
}


# 1 Mean/Mode Imputation (deterministic)
rec_mean_mode <- recipe(class ~., data = df) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

df_mean_mode <- prep(rec_mean_mode) %>% bake(new_data = NULL)

# 2 Regresion Imputation (deterministic)
preds_only <- df %>% select(-class)
imp_reg <- mice(preds_only, m = 1, method = "norm.predict", maxit = 20, printFlag = FALSE)
preds_reg <- complete(imp_reg, 1)
df_reg <- bind_cols(preds_reg, df["class"])

# 3 Regression with perturbation (stochastic)
imp_reg_pert <- mice(preds_only, m = 1, method = "norm", maxit = 20, printFlag = FALSE)
preds_reg_pert <- complete(imp_reg_pert, 1)
df_reg_pert <- bind_cols(preds_reg_pert, df["class"])


# 4.2 Drop Rows with missing values
df_complete <- df %>% drop_na()

# 4.3 add NA flags, then impute numerics
rec_missing_ind <- recipe(class ~., data = df) %>%
  step_indicate_na(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors())

df_missing_ind <- prep(rec_missing_ind) %>% bake(new_data = NULL)


# Train & Evaluate all sets
results <- bind_rows(
  fit_score(df_mean_mode, "Mean/Mode"),
  fit_score(df_reg, "Regression (deterministic)"),
  fit_score(df_reg_pert, "Regression + perturbation"),
  fit_score(df_complete, "Drop Missing-Case"),
  fit_score(df_missing_ind, "Missing-Indicator")
)

results %>%
  arrange(desc(ROC_AUC)) %>%
  mutate(across(c(Accuracy, F1, ROC_AUC), ~ round(.x, 4))) %>%
  print(n = Inf)














