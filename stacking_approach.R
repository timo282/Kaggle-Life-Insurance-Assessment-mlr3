library(mlr3)
library(mlr3verse)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3mbo)
library(Metrics)
library(future)
library(future.apply)
source("scripts/utils.R")
source("scripts/qwk_measure.R")

set.seed(123)

# ---- Load the data ----
data <- read.csv("data\\train.csv")
task <- as_task_classif(data, target = "Response")

# --- Classification ---
data_submission <- read.csv("data\\test.csv")
source("scripts/preprocessing.R")

xgboost_classif <- lrn("classif.xgboost", nrounds=972, eta=0.06007764, max_depth=6, colsample_bytree= 0.6227221, subsample=  0.8157236) 
xgboost_classif$predict_type = "prob"
learner_classif <- pipeline %>>% po_info_gain %>>% po("learner_cv", learner = xgboost_classif, resampling.method = "cv", resampling.folds = 3)

task_train_classif <- task$clone(deep = TRUE)$filter(split$train)
task_test_classif <- task$clone(deep = TRUE)$filter(split$test)

learner_classif$train(task_train_classif)
preds_classif <- learner_classif$predict(task_train_classif)
preds_classif_test <- learner_classif$predict(task_test_classif)

# --- Regression ---
data_submission <- read.csv("data\\test.csv")
task <- as_task_regr(data, target = "Response")

source("scripts/preprocessing.R")

xgboost_regr <- lrn("regr.xgboost", nrounds=972, eta=0.06007764, max_depth=6, colsample_bytree= 0.6227221, subsample=  0.8157236)
learner_regr <- pipeline %>>% po_info_gain %>>% po("learner_cv", learner = xgboost_regr, resampling.method = "cv", resampling.folds = 3)

task_train_regr <- task$clone(deep = TRUE)$filter(split$train)
task_test_regr <- task$clone(deep = TRUE)$filter(split$test)

learner_regr$train(task_train_regr)
preds_regr <- learner_regr$predict(task_train_regr)
preds_regr_test <- learner_regr$predict(task_test_regr)


# ---- Stacking ----
preds_combined <- cbind(preds_classif[[1]]$data(, paste0("classif.xgboost.prob.", 1:8)), preds_regr[[1]]$data(, "regr.xgboost.response"), preds_classif[[1]]$truth())
colnames(preds_combined) <- c("prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6", "prob_7", "prob_8", "regr", "truth")

preds_combined_test <- cbind(preds_classif_test[[1]]$data(, paste0("classif.xgboost.prob.", 1:8)), preds_regr_test[[1]]$data(, "regr.xgboost.response"), preds_classif_test[[1]]$truth())
colnames(preds_combined_test) <- c("prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6", "prob_7", "prob_8", "regr", "truth")

task_stacked <- as_task_classif(preds_combined, target = "truth")
task_stacked_test <- as_task_classif(preds_combined_test, target = "truth")

# ---- Train the learner ----
learner_level1 <- lrn("classif.rpart")
learner_level1$train(task_stacked)
preds_level1 <- learner_level1$predict(task_stacked_test)

measure <- MeasureClassifQuadraticWeightedKappa$new()

score <- measure$score(preds_level1)
print(score)

# --- Prepare the submission data ---
data_submission$Response = sample(1:8, nrow(data_submission), replace=TRUE)
task_submission_classif <- as_task_classif(data_submission, target = "Response")
task_submission_regr <- as_task_regr(data_submission, target = "Response")

preds_classif_submission <- learner_classif$predict(task_submission_classif)
preds_regr_submission <- learner_regr$predict(task_submission_regr)

preds_combined_submission <- cbind(preds_classif_submission[[1]]$data(, paste0("classif.xgboost.prob.", 1:8)), preds_regr_submission[[1]]$data(, "regr.xgboost.response"))
colnames(preds_combined_submission) <- c("prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6", "prob_7", "prob_8", "regr")


results <- cbind(preds_combined_submission, Id = data_submission$Id)

# ---- Submission ----
filename <- create_learner_submission(learner_level1, results, name = "xgboost_stacked")
print(filename)


