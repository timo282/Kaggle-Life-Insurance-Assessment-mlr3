write_results_with_timestamp <- function(results) {
  if (!dir.exists("submissions")) {
    dir.create("submissions")
  }

  current_datetime <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")

  filename <- paste0("submissions/submission_", current_datetime, ".csv")

  write.csv(results, filename, row.names = FALSE)

  return(filename)
}
