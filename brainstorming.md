- DONE remove columns with many missing values (>XX%) or fill missing values
    - Employment_Info_1      0.0003
    - Employment_Info_4      0.1142
    - Employment_Info_6      0.1828
    - Insurance_History_5    0.4277
    - Family_Hist_2          0.4826
    - Family_Hist_3          0.5766
    - Family_Hist_4          0.3231
    - Family_Hist_5          0.7041 remove
    - Medical_History_1      0.1497 fill?
    - Medical_History_10     0.9906 remove
    - Medical_History_15     0.7510 remove
    - Medical_History_24     0.9360 remove
    - Medical_History_32     0.9814 remove

- DONE remove id (unique)
- DONE Product_Info_1: encode binary?
- DONE Product_Info_2: check if ordinal scale? --> encoded ordinal
- DONE Product_Info_5: encode binary?
- DONE Product_Info_6: encode binary?
- Product_Info_7: encode OHE - what do we do here @moritz?
- Employment_Info_3: encode binary?
- Employment_Info_4: problem heavy tail!
- Employment_Info_5: encode binary
- Medical_History_3-9, Medical_History_11-14, Medical_History_16-23, Medical_History_25-31, Medical_History_33-41: binary or 3 - can this be aggregated somehow?
- Medical_Keyword_1-48 are all binary (0-1): construct new feature sum keywords and then perform pca and include first seven

- approaches: aggregate variables into one and encode differently / perform dimesion reduction OR perform feature selection (e.g. based on correlation)

- problem class imbalance - maybe address with class weights during training?

- models: pipeline with xgboost and random forest (tune or ensemble)
- use train-test-split for outer validation, cv5 for inner validation/tuning

- uses QWK for tuning and validation

- ohe or target/impact encoding for categorical features (fix or tune between two/three global options)

- feature selection: use mutual information or greedy selection - maybe also tune this?

ideas to predict output:
- regressor with tuned thesholds (boring)
- train a regressor and then stack a classifier on the results
- train a regressor and a probability classifier and stack a classifier on the results
