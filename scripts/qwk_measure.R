# Load necessary libraries
library(mlr3)
library(R6)

# Define the custom measure class
MeasureClassifQuadraticWeightedKappa = R6::R6Class("MeasureRegrQuadraticWeightedKappa",
  inherit = mlr3::MeasureClassif, # regression measure
  public = list(
    initialize = function() { # initialize class
      super$initialize(
        id = "quadratic_weighted_kappa", # unique ID
        packages = "Metrics", # package dependency
        properties = character(), # no special properties
        predict_type = "response", # measures response prediction
        range = c(-1, 1), # results in values between (-1, 1)
        minimize = FALSE # larger values are better
      )
    }
  ),

  private = list(
    # define score as private method
    .score = function(prediction, ...) {
      truth_int = as.integer(as.factor(prediction$truth))
      response_int = as.integer(as.factor(prediction$response))
      # define loss
      quadratic_weighted_kappa = function(truth, response) {
        Metrics::ScoreQuadraticWeightedKappa(truth, response)
      }
      # call loss function
      quadratic_weighted_kappa(truth_int, response_int)
    }
  )
)


MeasureRegrQuadraticWeightedKappa = R6::R6Class("MeasureRegrQuadraticWeightedKappa",
  inherit = mlr3::MeasureRegr, # regression measure
  public = list(
    initialize = function() { # initialize class
      super$initialize(
        id = "quadratic_weighted_kappa_regr", # unique ID
        packages = "Metrics", # package dependency
        properties = character(), # no special properties
        predict_type = "response", # measures response prediction
        range = c(-1, 1), # results in values between (-1, 1)
        minimize = FALSE # larger values are better
      )
    }
  ),

  private = list(
    # define score as private method
    .score = function(prediction, ...) {
      truth_int = as.integer(as.factor(prediction$truth))
      response_int = as.integer(as.factor(prediction$response))
      # define loss
      quadratic_weighted_kappa = function(truth, response) {
        Metrics::ScoreQuadraticWeightedKappa(truth, response)
      }
      # call loss function
      quadratic_weighted_kappa(truth_int, response_int)
    }
  )
)