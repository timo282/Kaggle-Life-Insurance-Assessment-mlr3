library(mlr3)
library(mlr3verse)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(Metrics)
library(future)
library(future.apply)
source("scripts/utils.R")
source("scripts/qwk_measure.R")

set.seed(123)

# ---- Load the data ----
data <- read.csv("data\\train.csv")
data_submission <- read.csv("data\\test.csv")

# create a mlr3 task
task <- as_task_classif(data, target = "Response")

# split the data into training and testing sets
split <- partition(task, ratio = 0.8)

# ---- Feature engineering ----
# for Medical_Keyword_1-48 all binary (0-1): construct new feature sum keywords
features_medical_keywords <- paste0("Medical_Keyword_", 1:48)

sum_columns <- function(data, columns) {
    if (length(columns) == 0) {
        stop("No columns to sum.")
    }
    data <- as.data.table(data)
    data.table(sum_medical_keywords = rowSums(data[, ..columns]))
}

sum_medical_keywords <- sum_columns(task$data(), features_medical_keywords)
task$cbind(sum_medical_keywords)

sum_medical_keywords_submission <- sum_columns(data_submission, features_medical_keywords)
data_submission <- cbind(data_submission, sum_medical_keywords_submission)

# ---- Create tasks ----
data_train <- task$data(rows = split$train)

task_train <- task$clone(deep = TRUE)$filter(split$train)
task_test <- task$clone(deep = TRUE)$filter(split$test)

# ---- Preprocessing ----

# -- missing value imputation  --

# remove features with more than 30% missing values
threshold <- 0.3
missing_features_rmv <- colnames(data_train)[colMeans(is.na(data_train)) > threshold]
selected_features <- setdiff(colnames(data_train), append(missing_features_rmv, "Id"))

# impute missing values
features_impute_mean <- c("Employment_Info_1", "Employment_Info_4", "Employment_Info_6")
features_imput_mode <- c("Medical_History_1")
po_select <- po("select", selector = selector_name(selected_features))
po_impute <- po(
    "imputemean",
    affect_columns = selector_name(features_impute_mean)
) %>>% po(
    "imputemode",
    affect_columns = selector_name(features_imput_mode)
)

# -- encode categorical features --

# ordinal encoding
features_encode_ordinal <- c("Product_Info_2")
ordinal_encode <- function(x) {
    as.integer(as.factor(x))
}
po_ordinal <- po("colapply",
    id = "ordinal_encode",
    applicator = ordinal_encode,
    affect_columns = selector_name(features_encode_ordinal)
)

# binary encoding
features_encode_binary <- c("Product_Info_1", "Product_Info_5", "Product_Info_6", "Employment_Info_3", "Employment_Info_5", "InsuredInfo_2", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", "Insurance_History_1")
binary_encode <- function(x) {
    x <- as.factor(x)
    levels_x <- levels(x)
    if (length(levels_x) != 2) stop("The variable must have exactly two levels")
    encoded <- as.integer(x) - 1
    if (!all(encoded %in% c(0, 1))) stop("Resulting values are not all 0 or 1")
    encoded
}
po_binary <- po("colapply",
    id = "binary_encode",
    applicator = binary_encode,
    affect_columns = selector_name(features_encode_binary)
)

# impact encoding
features_encode_impact <- c("Product_Info_7", "InsuredInfo_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", "Insurance_History_8", "Insurance_History_9", "Family_Hist_1")
convert_to_factors <- function(x) {
    as.factor(x)
}
po_impact <- po("colapply",
    id = "convert_to_factors",
    applicator = convert_to_factors,
    affect_columns = selector_name(features_encode_impact)
) %>>% po("encodeimpact", affect_columns = selector_name(features_encode_impact))


# -- Special encodings --
# perform a pca on the medical keywords 1-48, selecting the first 7 pcs
features_pca <- paste0("Medical_Keyword_", 1:48)
po_pca <- po("pca", id="pca_medical_keyword", affect_columns = selector_name(features_pca), rank. = 7)

# remove the original medical keywords
po_remove_medical_keywords <- po("select", id = "remove_medical_keywords", selector = selector_invert(selector_name(features_pca)))

# rename the pca columns
name_mapping <- setNames(paste0("Medical_Keyword_PC", 1:7), paste0("PC", 1:7))
po_rename_pca <- po("renamecolumns", id="rename_keyword_pcs", renaming = name_mapping)

po_medical_keywords <- po_pca %>>% po_remove_medical_keywords %>>% po_rename_pca

# for medical history 1-41, impact encode columns 3-41 and then perform pca on all 41
features_medical_history <- paste0("Medical_History_", 3:41)
po_medical_history_impact <- po("encodeimpact", id="impact_enc_medical_history", affect_columns = selector_name(features_medical_history))
po_medical_history_scale <- po("scalerange", id="scale_medical_history", affect_columns = selector_name(paste0("Medical_History_", 1:41)))
po_medical_history_pca <- po("pca", id="pca_medical_history", affect_columns = selector_name(paste0("Medical_History_", 1:41)), rank. = 10)
po_remove_medical_history <- po("select", id = "remove_medical_history", selector = selector_invert(selector_name(paste0("Medical_History_", 1:41))))
name_mapping <- setNames(paste0("Medical_History_PC", 1:10), paste0("PC", 1:10))
po_rename_history_pca <- po("renamecolumns", id="rename_history_pcs", renaming = name_mapping)

po_medical_history <- po_medical_history_impact %>>% po_medical_history_scale %>>% po_medical_history_pca %>>% po_remove_medical_history %>>% po_rename_history_pca

# -- Scaling --
po_scale <- po("scalerange")

# -- Combine all preprocessing steps --
pipeline <- po_select %>>% po_impute %>>% po_binary %>>% po_ordinal %>>% po_impact %>>% po_medical_keywords %>>% po_medical_history %>>% po_scale

# ---- Feature selection ----
# feature selection using information gain
po_info_gain <- po("filter", filter = flt("information_gain"), filter.nfeat = 25)

# ---- Learner ----
xgboost <- lrn("classif.xgboost")
learner <- as_learner(pipeline %>>% po_info_gain %>>% xgboost)

# ---- Tuning ----

# Define the search space
param_set <- ParamSet$new(list(
  ParamInt$new("classif.xgboost.nrounds", lower = 50, upper = 1000),
  ParamDbl$new("classif.xgboost.eta", lower = 0.01, upper = 0.3),
  ParamInt$new("classif.xgboost.max_depth", lower = 3, upper = 20),
  ParamDbl$new("classif.xgboost.colsample_bytree", lower = 0.5, upper = 1.0),
  ParamDbl$new("classif.xgboost.subsample", lower = 0.5, upper = 1.0),
  ParamInt$new("information_gain.filter.nfeat", lower = 10, upper = 50)
))

# Define the tuner
tuner <- tnr("mbo")

measure <- MeasureClassifQuadraticWeightedKappa$new()

# Define the AutoTuner
at <- AutoTuner$new(
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measure = measure,
  terminator = trm("evals", n_evals = 50),
  tuner = tuner,
  search_space = param_set
)

# ---- Training with tuning ----
plan(multisession)
at$train(task_train)
plan(sequential)

# ---- Prediction ----
preds <- at$predict(task_test)

score <- measure$score(preds)
print(score)

# ---- Submission ----
filename <- create_learner_submission(at$learner, data_submission, name = "xgboost")
print(filename)

