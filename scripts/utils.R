library(mlr3)
library(mlr3pipelines)
library(R6)


create_learner_submission <- function(learner, data_test, name="") {
    if (!dir.exists("submissions")) {
        dir.create("submissions")
    }
    current_datetime <- format(Sys.time(), "%Y%m%d%H%M")
    filename <- paste0("submissions/submission_", name, current_datetime, ".csv")

    preds <- learner$predict_newdata(newdata = data_test)
    results <- data.frame(Id = data_test$Id, Response = as.integer(preds$response))

    write.csv(results, filename, row.names = FALSE)

    return(filename)
}


PipeOpRegrToClassif <- R6::R6Class("PipeOpRegrToClassif",
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  public = list(
    initialize = function(id = "regr_to_classif", param_vals = list()) {
      super$initialize(id, param_vals = param_vals, packages = "mlr3")
    }
  ),
  private = list(
    .train_task = function(task) {
      data <- task$data()
      target_name <- task$target_names
      
      # Ensure the target variable is treated as a factor (multiclass)
      data[[target_name]] <- as.factor(data[[target_name]])
      
      # Create a new classification task
      task_classif <- as_task_classif(data, target = target_name)
      return(task_classif)
    },
    .predict_task = function(task) {
      task
    }
  )
)

PipeOpClassifToRegr <- R6::R6Class("PipeOpClassifToRegr",
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  public = list(
    initialize = function(id = "classif_to_regr", param_vals = list()) {
      super$initialize(id, param_vals = param_vals, packages = "mlr3")
    }
  ),
  private = list(
    .train_task = function(task) {
      data <- task$data()
      target_name <- task$target_names
      
      # Convert the target variable to numeric
      data[[target_name]] <- as.numeric(as.character(data[[target_name]]))
      
      # Create a new regression task
      task_regr <- as_task_regr(data, target = target_name)
      return(task_regr)
    },
    .predict_task = function(task) {
      task
    }
  )
)

