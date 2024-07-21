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
