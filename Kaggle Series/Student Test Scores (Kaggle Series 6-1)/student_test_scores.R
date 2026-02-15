# ============================================================================
# Student Test Scores - R Translation
# Literal translation from Python notebook
# ============================================================================

# Load libraries
library(tidyverse)      # dplyr, ggplot2, readr, etc.
library(caret)          # For modeling utilities
library(xgboost)        # XGBoost
library(lightgbm)       # LightGBM
library(catboost)       # CatBoost
library(glmnet)         # Ridge/Lasso regression
library(Metrics)        # For RMSE calculation

# ============================================================================
# EDA
# ============================================================================

# Read data
df <- read_csv('train.csv')
test_df <- read_csv('test.csv')

# Shape
cat("Train shape:", nrow(df), "x", ncol(df), "\n")

# Data types
str(df)

# Unique classes for categorical columns
cat_cols <- df %>% select(where(is.character)) %>% names()
for (col in cat_cols) {
  cat(col, ":", n_distinct(df[[col]]), "unique classes\n")
}

# Not too many unique classes, which is nice when training a model

# Missing values
colSums(is.na(df))

# Duplicates
sum(duplicated(df))

# ============================================================================
# Subplots - Boxplots for categorical variables
# ============================================================================

# Get categorical columns
categorical_features <- df %>% select(where(is.character)) %>% names()

# Create boxplots for each categorical feature
boxplot_list <- map(categorical_features, function(feature) {
  ggplot(df, aes(x = .data[[feature]], y = exam_score)) +
    geom_boxplot(fill = "steelblue", alpha = 0.7) +
    labs(title = paste("Exam Score by", feature)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})

# Print boxplots (in RStudio, you can view them one by one)
# Or use gridExtra to arrange them
library(gridExtra)
do.call(grid.arrange, c(boxplot_list, ncol = 3))

# ============================================================================
# Line plots for numerical variables
# ============================================================================

# Get numerical columns (excluding id and exam_score)
numerical_features <- df %>% 
  select(where(is.numeric)) %>% 
  select(-id, -exam_score) %>% 
  names()

# Create line plots for each numerical feature
lineplot_list <- map(numerical_features, function(feature) {
  grouped_data <- df %>%
    group_by(.data[[feature]]) %>%
    summarise(mean_score = mean(exam_score), .groups = 'drop')
  
  ggplot(grouped_data, aes(x = .data[[feature]], y = mean_score)) +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_point(color = "steelblue", size = 2) +
    labs(title = paste("Average Exam Score by", feature),
         x = feature,
         y = "Average Exam Score") +
    theme_minimal()
})

do.call(grid.arrange, c(lineplot_list, ncol = 2))

# I see solid results, in the categorical columns you can see not too much 
# differences between categories, being the ones with more differences the 
# ones you could expect (sleep_quality, facility_rating, study_method).
# For numerical columns, we can see clear tendencies and some outliers.

# ============================================================================
# Outliers
# ============================================================================

# Outlier detection using IQR method
detect_outliers_iqr <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25)
  Q3 <- quantile(data[[column]], 0.75)
  IQR_val <- Q3 - Q1
  
  lower_bound <- Q1 - 1.5 * IQR_val
  upper_bound <- Q3 + 1.5 * IQR_val
  
  outliers <- data %>% filter(.data[[column]] < lower_bound | .data[[column]] > upper_bound)
  
  return(list(
    outliers = outliers,
    lower_bound = lower_bound,
    upper_bound = upper_bound,
    Q1 = Q1,
    Q3 = Q3,
    IQR = IQR_val
  ))
}

# Get all numerical columns
numerical_cols <- df %>% select(where(is.numeric), -id) %>% names()

cat("OUTLIER ANALYSIS USING IQR METHOD\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

outlier_summary <- list()

for (col in numerical_cols) {
  result <- detect_outliers_iqr(df, col)
  
  cat("\n", toupper(col), ":\n", sep = "")
  cat("  Q1:", sprintf("%.2f", result$Q1), "\n")
  cat("  Q3:", sprintf("%.2f", result$Q3), "\n")
  cat("  IQR:", sprintf("%.2f", result$IQR), "\n")
  cat("  Lower bound:", sprintf("%.2f", result$lower_bound), "\n")
  cat("  Upper bound:", sprintf("%.2f", result$upper_bound), "\n")
  cat("  Number of outliers:", nrow(result$outliers), "\n")
  cat("  Percentage of outliers:", sprintf("%.2f%%", (nrow(result$outliers)/nrow(df))*100), "\n")
  
  if (nrow(result$outliers) > 0) {
    cat("  Outlier range:", sprintf("%.2f", min(result$outliers[[col]])), 
        "to", sprintf("%.2f", max(result$outliers[[col]])), "\n")
  }
  
  outlier_summary[[col]] <- list(
    count = nrow(result$outliers),
    percentage = (nrow(result$outliers)/nrow(df))*100,
    lower_bound = result$lower_bound,
    upper_bound = result$upper_bound
  )
}

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("SUMMARY:\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
for (col in names(outlier_summary)) {
  cat(col, ":", outlier_summary[[col]]$count, "outliers", 
      sprintf("(%.2f%%)", outlier_summary[[col]]$percentage), "\n")
}

# ============================================================================
# Target Distribution
# ============================================================================

# Histogram and boxplot of exam_score
p1 <- ggplot(df, aes(x = exam_score)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "steelblue", alpha = 0.7) +
  geom_density(color = "darkblue", linewidth = 1) +
  labs(title = "Distribution of Exam Score") +
  theme_minimal()

p2 <- ggplot(df, aes(x = exam_score)) +
  geom_boxplot(fill = "steelblue", alpha = 0.7) +
  labs(title = "Boxplot of Exam Score") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

summary(df$exam_score)

# ============================================================================
# Correlation
# ============================================================================

# Correlation matrix for numerical columns
num_cols_for_corr <- df %>% select(where(is.numeric), -id) %>% names()
cor_matrix <- cor(df[num_cols_for_corr])

# Heatmap using ggplot2
library(reshape2)
cor_melted <- melt(cor_matrix)

ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", value)), color = "white", size = 3) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Matrix") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ============================================================================
# Train vs Test Distribution
# ============================================================================

cat("Train shape:", nrow(df), "x", ncol(df), "\n")
cat("Test shape:", nrow(test_df), "x", ncol(test_df), "\n")

# Check distributions are similar
num_features <- df %>% select(where(is.numeric), -id, -exam_score) %>% names()
for (col in num_features) {
  cat("\n", col, ":\n", sep = "")
  cat("  Train mean:", sprintf("%.2f", mean(df[[col]])), 
      ", Test mean:", sprintf("%.2f", mean(test_df[[col]])), "\n")
}

# ============================================================================
# EDA Conclusions
# 
# All data looks clean and ready for modeling:
# - No outliers detected using IQR method
# - No missing values or duplicates
# - Balanced distributions between train and test sets
# - Target variable (exam_score) follows a near-normal distribution
# - Low multicollinearity between features
#
# Preprocessing needed:
# - Categorical encoding for 7 object columns (will be handled during model pipeline)
# ============================================================================

# ============================================================================
# MODELING
# ============================================================================

# ============================================================================
# Data Preparation
# ============================================================================

# Separate features and target
target <- "exam_score"
features <- setdiff(names(df), c("id", target))

X <- df[features]
y <- df[[target]]
X_test <- test_df[features]

# Identify categorical and numerical columns
cat_cols <- X %>% select(where(is.character)) %>% names()
num_cols <- X %>% select(where(is.numeric)) %>% names()

cat("Features:", length(features), "\n")
cat("Categorical:", paste(cat_cols, collapse = ", "), "\n")
cat("Numerical:", paste(num_cols, collapse = ", "), "\n")
cat("\nTrain shape:", nrow(X), "x", ncol(X), "\n")
cat("Test shape:", nrow(X_test), "x", ncol(X_test), "\n")

# ============================================================================
# Target Encoding for categorical columns
# ============================================================================

# Custom target encoding function
target_encode <- function(train_data, test_data, cat_cols, target_col, smoothing = 10) {
  train_encoded <- train_data
  test_encoded <- test_data
  global_mean <- mean(target_col)
  
  for (col in cat_cols) {
    # Calculate target statistics per category
    stats <- train_data %>%
      mutate(target = target_col) %>%
      group_by(.data[[col]]) %>%
      summarise(
        cat_mean = mean(target),
        cat_count = n(),
        .groups = 'drop'
      ) %>%
      mutate(
        # Smoothed mean: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
        encoded = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
      )
    
    # Create lookup table
    encoding_map <- setNames(stats$encoded, stats[[col]])
    
    # Apply encoding
    train_encoded[[col]] <- encoding_map[train_data[[col]]]
    test_encoded[[col]] <- encoding_map[test_data[[col]]]
    
    # Handle unseen categories in test with global mean
    test_encoded[[col]][is.na(test_encoded[[col]])] <- global_mean
  }
  
  return(list(train = train_encoded, test = test_encoded))
}

encoded <- target_encode(X, X_test, cat_cols, y)
X_encoded <- encoded$train
X_test_encoded <- encoded$test

cat("Target Encoding complete!\n")
cat("Encoded", length(cat_cols), "categorical columns\n")
head(X_encoded)

# Convert to numeric matrices for modeling
X_encoded <- X_encoded %>% mutate(across(everything(), as.numeric))
X_test_encoded <- X_test_encoded %>% mutate(across(everything(), as.numeric))

# ============================================================================
# Cross-validation setup
# ============================================================================

N_FOLDS <- 5
set.seed(42)
folds <- createFolds(y, k = N_FOLDS, list = TRUE, returnTrain = FALSE)

# Function to evaluate model using K-Fold cross-validation
evaluate_model <- function(train_func, X, y, X_test, model_name = "Model") {
  oof_preds <- rep(0, nrow(X))
  test_preds <- rep(0, nrow(X_test))
  scores <- c()
  
  for (fold_idx in 1:N_FOLDS) {
    val_idx <- folds[[fold_idx]]
    train_idx <- setdiff(1:nrow(X), val_idx)
    
    X_train <- X[train_idx, ]
    X_val <- X[val_idx, ]
    y_train <- y[train_idx]
    y_val <- y[val_idx]
    
    # Train model and get predictions
    result <- train_func(X_train, y_train, X_val, X_test)
    
    oof_preds[val_idx] <- result$val_preds
    test_preds <- test_preds + result$test_preds / N_FOLDS
    
    fold_rmse <- rmse(y_val, oof_preds[val_idx])
    scores <- c(scores, fold_rmse)
    cat(sprintf("  Fold %d: RMSE = %.5f\n", fold_idx, fold_rmse))
  }
  
  overall_rmse <- rmse(y, oof_preds)
  cat(sprintf("\n  %s Overall CV RMSE: %.5f\n", model_name, overall_rmse))
  
  return(list(oof = oof_preds, test = test_preds, rmse = overall_rmse))
}

# ============================================================================
# Baseline Model (Ridge)
# ============================================================================

cat("Training Ridge (Baseline)...\n")

ridge_train <- function(X_train, y_train, X_val, X_test) {
  # Convert to matrix format
  X_train_mat <- as.matrix(X_train)
  X_val_mat <- as.matrix(X_val)
  X_test_mat <- as.matrix(X_test)
  
  # Fit Ridge regression (alpha = 0 in glmnet is Ridge)
  model <- glmnet(X_train_mat, y_train, alpha = 0, lambda = 1.0)
  
  val_preds <- predict(model, X_val_mat, s = 1.0)[,1]
  test_preds <- predict(model, X_test_mat, s = 1.0)[,1]
  
  return(list(val_preds = val_preds, test_preds = test_preds))
}

ridge_result <- evaluate_model(ridge_train, X_encoded, y, X_test_encoded, "Ridge")
ridge_oof <- ridge_result$oof
ridge_test <- ridge_result$test
ridge_rmse <- ridge_result$rmse

# ============================================================================
# Hyperparameter Tuning (using grid search since Optuna isn't available in R)
# For a more similar approach, you could use mlr3tuning package
# ============================================================================

cat("\nHyperparameter tuning for XGBoost using grid search...\n")
cat("This may take several minutes...\n")

# Define parameter grid (simplified version)
param_grid <- expand.grid(
  max_depth = c(4, 5, 6),
  eta = c(0.05, 0.08, 0.1),
  subsample = c(0.8, 0.9),
  colsample_bytree = c(0.7, 0.8),
  min_child_weight = c(5, 6)
)

# Quick 3-fold CV for tuning
set.seed(42)
tune_folds <- createFolds(y, k = 3, list = TRUE, returnTrain = FALSE)

best_rmse <- Inf
best_params <- NULL

# Sample a subset of grid for faster execution
set.seed(42)
sample_grid <- param_grid[sample(1:nrow(param_grid), min(15, nrow(param_grid))), ]

for (i in 1:nrow(sample_grid)) {
  params <- as.list(sample_grid[i, ])
  
  fold_scores <- c()
  for (fold_idx in 1:3) {
    val_idx <- tune_folds[[fold_idx]]
    train_idx <- setdiff(1:nrow(X_encoded), val_idx)
    
    dtrain <- xgb.DMatrix(data = as.matrix(X_encoded[train_idx, ]), label = y[train_idx])
    dval <- xgb.DMatrix(data = as.matrix(X_encoded[val_idx, ]), label = y[val_idx])
    
    xgb_params <- list(
      objective = "reg:squarederror",
      max_depth = params$max_depth,
      eta = params$eta,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      min_child_weight = params$min_child_weight
    )
    
    model <- xgb.train(
      params = xgb_params,
      data = dtrain,
      nrounds = 500,
      watchlist = list(val = dval),
      early_stopping_rounds = 50,
      verbose = 0
    )
    
    preds <- predict(model, dval)
    fold_scores <- c(fold_scores, rmse(y[val_idx], preds))
  }
  
  mean_rmse <- mean(fold_scores)
  if (mean_rmse < best_rmse) {
    best_rmse <- mean_rmse
    best_params <- params
    cat(sprintf("Trial %d: RMSE = %.5f (new best)\n", i, mean_rmse))
  }
}

cat(sprintf("\nBest trial RMSE: %.5f\n", best_rmse))
cat("\nBest hyperparameters:\n")
print(best_params)

# ============================================================================
# XGBoost with optimized hyperparameters
# ============================================================================

cat("\nTraining XGBoost with optimized hyperparameters...\n")

xgb_train <- function(X_train, y_train, X_val, X_test, params = best_params) {
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dval <- xgb.DMatrix(data = as.matrix(X_val))
  dtest <- xgb.DMatrix(data = as.matrix(X_test))
  
  xgb_params <- list(
    objective = "reg:squarederror",
    max_depth = params$max_depth,
    eta = params$eta,
    subsample = params$subsample,
    colsample_bytree = params$colsample_bytree,
    min_child_weight = params$min_child_weight
  )
  
  model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 1000,
    verbose = 0
  )
  
  val_preds <- predict(model, dval)
  test_preds <- predict(model, dtest)
  
  return(list(val_preds = val_preds, test_preds = test_preds))
}

xgb_result <- evaluate_model(xgb_train, X_encoded, y, X_test_encoded, "XGBoost (Optimized)")
xgb_oof <- xgb_result$oof
xgb_test <- xgb_result$test
xgb_rmse <- xgb_result$rmse

# ============================================================================
# LightGBM
# ============================================================================

cat("\nTraining LightGBM...\n")

lgb_train <- function(X_train, y_train, X_val, X_test) {
  dtrain <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
  
  params <- list(
    objective = "regression",
    metric = "rmse",
    num_leaves = 63,  # equivalent to max_depth ~6
    learning_rate = 0.05,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 1,
    verbose = -1
  )
  
  model <- lgb.train(
    params = params,
    data = dtrain,
    nrounds = 500
  )
  
  val_preds <- predict(model, as.matrix(X_val))
  test_preds <- predict(model, as.matrix(X_test))
  
  return(list(val_preds = val_preds, test_preds = test_preds))
}

lgb_result <- evaluate_model(lgb_train, X_encoded, y, X_test_encoded, "LightGBM")
lgb_oof <- lgb_result$oof
lgb_test <- lgb_result$test
lgb_rmse <- lgb_result$rmse

# ============================================================================
# CatBoost (can handle categorical features natively)
# ============================================================================

cat("\nTraining CatBoost...\n")

# For CatBoost, use original data with categorical features
X_cat <- df[features]
X_test_cat <- test_df[features]

# Get categorical column indices (0-based for catboost)
cat_feature_indices <- which(names(X_cat) %in% cat_cols) - 1

catboost_train <- function(X_train, y_train, X_val, X_test) {
  # Convert character columns to factors
  X_train_cat <- X_train %>% mutate(across(where(is.character), as.factor))
  X_val_cat <- X_val %>% mutate(across(where(is.character), as.factor))
  X_test_cat <- X_test %>% mutate(across(where(is.character), as.factor))
  
  train_pool <- catboost.load_pool(data = X_train_cat, label = y_train, cat_features = cat_cols)
  val_pool <- catboost.load_pool(data = X_val_cat, cat_features = cat_cols)
  test_pool <- catboost.load_pool(data = X_test_cat, cat_features = cat_cols)
  
  params <- list(
    iterations = 500,
    depth = 6,
    learning_rate = 0.05,
    loss_function = "RMSE",
    random_seed = 42,
    verbose = 0
  )
  
  model <- catboost.train(train_pool, params = params)
  
  val_preds <- catboost.predict(model, val_pool)
  test_preds <- catboost.predict(model, test_pool)
  
  return(list(val_preds = val_preds, test_preds = test_preds))
}

# CatBoost needs special handling for categorical features
cat_oof <- rep(0, nrow(X_cat))
cat_test <- rep(0, nrow(X_test_cat))

for (fold_idx in 1:N_FOLDS) {
  val_idx <- folds[[fold_idx]]
  train_idx <- setdiff(1:nrow(X_cat), val_idx)
  
  result <- catboost_train(X_cat[train_idx, ], y[train_idx], X_cat[val_idx, ], X_test_cat)
  
  cat_oof[val_idx] <- result$val_preds
  cat_test <- cat_test + result$test_preds / N_FOLDS
  
  fold_rmse <- rmse(y[val_idx], cat_oof[val_idx])
  cat(sprintf("  Fold %d: RMSE = %.5f\n", fold_idx, fold_rmse))
}

cat_rmse <- rmse(y, cat_oof)
cat(sprintf("\n  CatBoost Overall CV RMSE: %.5f\n", cat_rmse))

# ============================================================================
# Model Comparison & Ensemble
# ============================================================================

results <- c(
  Ridge = ridge_rmse,
  XGBoost = xgb_rmse,
  LightGBM = lgb_rmse,
  CatBoost = cat_rmse
)

cat(paste(rep("=", 50), collapse = ""), "\n")
cat("MODEL COMPARISON (CV RMSE - lower is better)\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

for (model in names(sort(results))) {
  cat(sprintf("%-12s: %.5f\n", model, results[model]))
}

best_model <- names(which.min(results))
cat(sprintf("\nBest single model: %s (%.5f)\n", best_model, results[best_model]))

# ============================================================================
# Ensemble with optimized weights
# ============================================================================

# Simple average ensemble first
ensemble_oof_simple <- (xgb_oof + lgb_oof + cat_oof) / 3
ensemble_test_simple <- (xgb_test + lgb_test + cat_test) / 3
simple_rmse <- rmse(y, ensemble_oof_simple)
cat(sprintf("\nSimple Ensemble (1/3 each) CV RMSE: %.5f\n", simple_rmse))

# Optimize ensemble weights using optim
rmse_ensemble <- function(weights) {
  pred <- weights[1] * xgb_oof + weights[2] * lgb_oof + weights[3] * cat_oof
  return(rmse(y, pred))
}

# Constraint: weights must sum to 1
# Using L-BFGS-B with bounds
initial_weights <- c(0.33, 0.33, 0.34)

# Grid search for optimal weights (simple approach)
best_ensemble_rmse <- Inf
best_weights <- c(1/3, 1/3, 1/3)

for (w1 in seq(0, 1, by = 0.05)) {
  for (w2 in seq(0, 1 - w1, by = 0.05)) {
    w3 <- 1 - w1 - w2
    weights <- c(w1, w2, w3)
    current_rmse <- rmse_ensemble(weights)
    if (current_rmse < best_ensemble_rmse) {
      best_ensemble_rmse <- current_rmse
      best_weights <- weights
    }
  }
}

cat(sprintf("\nOptimal weights: XGB=%.3f, LGB=%.3f, CAT=%.3f\n", 
            best_weights[1], best_weights[2], best_weights[3]))

# Calculate optimized ensemble predictions
ensemble_oof_opt <- best_weights[1] * xgb_oof + best_weights[2] * lgb_oof + best_weights[3] * cat_oof
ensemble_test_opt <- best_weights[1] * xgb_test + best_weights[2] * lgb_test + best_weights[3] * cat_test
optimized_rmse <- rmse(y, ensemble_oof_opt)
cat(sprintf("Optimized Ensemble CV RMSE: %.5f\n", optimized_rmse))

# Add ensembles to results
results["Simple Ensemble"] <- simple_rmse
results["Optimized Ensemble"] <- optimized_rmse

# Find overall best
cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("FINAL COMPARISON (including ensembles)\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

for (model in names(sort(results))) {
  cat(sprintf("%-20s: %.5f\n", model, results[model]))
}

best_model <- names(which.min(results))
cat(sprintf("\nBest model: %s (%.5f)\n", best_model, results[best_model]))

# Select final predictions based on best model
if (best_model == "Optimized Ensemble") {
  final_predictions <- ensemble_test_opt
  final_model <- "Optimized Ensemble"
} else if (best_model == "Simple Ensemble") {
  final_predictions <- ensemble_test_simple
  final_model <- "Simple Ensemble"
} else if (best_model == "XGBoost") {
  final_predictions <- xgb_test
  final_model <- "XGBoost"
} else if (best_model == "LightGBM") {
  final_predictions <- lgb_test
  final_model <- "LightGBM"
} else if (best_model == "CatBoost") {
  final_predictions <- cat_test
  final_model <- "CatBoost"
} else {
  final_predictions <- ridge_test
  final_model <- "Ridge"
}

# ============================================================================
# Stacking (Meta-Model Approach)
# ============================================================================

cat("\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Stacking (Meta-Model Approach)\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

# Create stacking features from OOF predictions
stack_train <- cbind(xgb_oof, lgb_oof, cat_oof, ridge_oof)
stack_test <- cbind(xgb_test, lgb_test, cat_test, ridge_test)

cat(sprintf("Stacking features shape: %d x %d\n", nrow(stack_train), ncol(stack_train)))

# Train meta-model (Ridge with cross-validation for alpha)
# Find best alpha using cross-validation
alphas <- c(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
best_alpha <- NULL
best_cv_rmse <- Inf

for (alpha in alphas) {
  cv_scores <- c()
  for (fold_idx in 1:N_FOLDS) {
    val_idx <- folds[[fold_idx]]
    train_idx <- setdiff(1:nrow(stack_train), val_idx)
    
    model <- glmnet(stack_train[train_idx, ], y[train_idx], alpha = 0, lambda = alpha)
    preds <- predict(model, stack_train[val_idx, ], s = alpha)[,1]
    cv_scores <- c(cv_scores, rmse(y[val_idx], preds))
  }
  
  mean_cv_rmse <- mean(cv_scores)
  if (mean_cv_rmse < best_cv_rmse) {
    best_cv_rmse <- mean_cv_rmse
    best_alpha <- alpha
  }
}

cat(sprintf("Best alpha for meta-model: %.3f\n", best_alpha))

# Get stacking OOF predictions with proper cross-validation
stack_oof_preds <- rep(0, length(y))

for (fold_idx in 1:N_FOLDS) {
  val_idx <- folds[[fold_idx]]
  train_idx <- setdiff(1:nrow(stack_train), val_idx)
  
  fold_meta <- glmnet(stack_train[train_idx, ], y[train_idx], alpha = 0, lambda = best_alpha)
  stack_oof_preds[val_idx] <- predict(fold_meta, stack_train[val_idx, ], s = best_alpha)[,1]
}

# Final stacking test predictions
meta_model <- glmnet(stack_train, y, alpha = 0, lambda = best_alpha)
stack_test_preds <- predict(meta_model, stack_test, s = best_alpha)[,1]

# Calculate RMSE for stacking
stack_rmse <- rmse(y, stack_oof_preds)
cat(sprintf("\nStacking CV RMSE: %.5f\n", stack_rmse))

# Add to results
results["Stacking"] <- stack_rmse

# Update best model comparison
cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("FINAL COMPARISON (with Stacking)\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

for (model in names(sort(results))) {
  marker <- ifelse(results[model] == min(results), "*", " ")
  cat(sprintf("%s %-20s: %.5f\n", marker, model, results[model]))
}

# Update final predictions if stacking is best
best_overall <- names(which.min(results))
if (best_overall == "Stacking") {
  final_predictions <- stack_test_preds
  final_model <- "Stacking"
  cat("\nStacking is the new best model!\n")
}

# ============================================================================
# Generate Submission
# ============================================================================

submission <- data.frame(
  id = test_df$id,
  exam_score = final_predictions
)

write_csv(submission, 'submission_r.csv')
cat(sprintf("\nSubmission saved using: %s\n", final_model))
cat(sprintf("Shape: %d x %d\n", nrow(submission), ncol(submission)))
head(submission, 10)

# ============================================================================
# Sanity check: Compare train vs predicted distributions
# ============================================================================

p1 <- ggplot(data.frame(exam_score = y), aes(x = exam_score)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "steelblue", alpha = 0.7) +
  geom_density(color = "darkblue", linewidth = 1) +
  labs(title = "Train exam_score Distribution", x = "exam_score") +
  theme_minimal()

p2 <- ggplot(data.frame(exam_score = final_predictions), aes(x = exam_score)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "forestgreen", alpha = 0.7) +
  geom_density(color = "darkgreen", linewidth = 1) +
  labs(title = "Test Predictions Distribution", x = "exam_score") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

cat(sprintf("\nTrain stats:      mean=%.2f, std=%.2f\n", mean(y), sd(y)))
cat(sprintf("Prediction stats: mean=%.2f, std=%.2f\n", mean(final_predictions), sd(final_predictions)))

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("SCRIPT COMPLETED SUCCESSFULLY!\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

