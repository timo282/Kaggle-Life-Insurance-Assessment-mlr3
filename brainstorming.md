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
- DONE Employment_Info_3: encode binary?
- Employment_Info_4: problem heavy tail!
- DONE Employment_Info_5: encode binary
- Medical_History_3-9, Medical_History_11-14, Medical_History_16-23, Medical_History_25-31, Medical_History_33-41: binary or 3 - can this be aggregated somehow?
- DONE Medical_Keyword_1-48 are all binary (0-1): construct new feature sum keywords and then perform pca and include first seven

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



Check if all columns have been preprocessed:
- D 'Id', removed
- D 'Product_Info_1', binary enc
- D 'Product_Info_2', ordinal enc
- D 'Product_Info_3',
- D 'Product_Info_4',
- D 'Product_Info_5', binary enc
- D 'Product_Info_6', binary enc
- D 'Product_Info_7',
- D 'Ins_Age',
- D 'Ht',
- D 'Wt',
- D 'BMI',
- D 'Employment_Info_1', impute mean
- D 'Employment_Info_2',
- D 'Employment_Info_3', binary enc
- D 'Employment_Info_4', impute mean
- D 'Employment_Info_5', binary enc
- D 'Employment_Info_6', impute mean
- D 'InsuredInfo_1', target enc
- D 'InsuredInfo_2', binary enc
- D 'InsuredInfo_3',
- D 'InsuredInfo_4', binary enc
- D 'InsuredInfo_5', binary enc
- D 'InsuredInfo_6', binary enc
- D 'InsuredInfo_7', binary enc
- D 'Insurance_History_1', binary enc
- D 'Insurance_History_2', target enc
- D 'Insurance_History_3', target enc
- D 'Insurance_History_4', target enc
- D 'Insurance_History_5', removed
- D 'Insurance_History_7', target enc
- D 'Insurance_History_8', target enc
- D 'Insurance_History_9', target enc
- D 'Family_Hist_1', target enc
- D 'Family_Hist_2', removed
- D 'Family_Hist_3', removed
- D 'Family_Hist_4', removed
- D 'Family_Hist_5', removed
- D 'Medical_History_1', impute mode, target enc, pca
- D 'Medical_History_2', target enc, pca
- D 'Medical_History_3', target enc, pca
- D 'Medical_History_4', target enc, pca
- D 'Medical_History_5', target enc, pca
- D 'Medical_History_6', target enc, pca
- D 'Medical_History_7', target enc, pca
- D 'Medical_History_8', target enc, pca
- D 'Medical_History_9', target enc, pca
- D 'Medical_History_10', removed
- D 'Medical_History_11', target enc, pca
- D 'Medical_History_12', target enc, pca
- D 'Medical_History_13', target enc, pca
- D 'Medical_History_14', target enc, pca
- D 'Medical_History_15', removed
- D 'Medical_History_16', target enc, pca
- D 'Medical_History_17', target enc, pca
- D 'Medical_History_18', target enc, pca
- D 'Medical_History_19', target enc, pca
- D 'Medical_History_20', target enc, pca
- D 'Medical_History_21', target enc, pca
- D 'Medical_History_22', target enc, pca
- D 'Medical_History_23', target enc, pca
- D 'Medical_History_24', removed
- D 'Medical_History_25', target enc, pca
- D 'Medical_History_26', target enc, pca
- D 'Medical_History_27', target enc, pca
- D 'Medical_History_28', target enc, pca
- D 'Medical_History_29', target enc, pca
- D 'Medical_History_30', target enc, pca
- D 'Medical_History_31', target enc, pca
- D 'Medical_History_32', removed
- D 'Medical_History_33', target enc, pca
- D 'Medical_History_34', target enc, pca
- D 'Medical_History_35', target enc, pca
- D 'Medical_History_36', target enc, pca
- D 'Medical_History_37', target enc, pca
- D 'Medical_History_38', target enc, pca
- D 'Medical_History_39', target enc, pca
- D 'Medical_History_40', target enc, pca
- D 'Medical_History_41', target enc, pca
- D 'Medical_Keyword_1', summed and PCA
- D 'Medical_Keyword_2', summed and PCA
- D 'Medical_Keyword_3', summed and PCA
- D 'Medical_Keyword_4', summed and PCA
- D 'Medical_Keyword_5', summed and PCA
- D 'Medical_Keyword_6', summed and PCA
- D 'Medical_Keyword_7', summed and PCA
- D 'Medical_Keyword_8', summed and PCA
- D 'Medical_Keyword_9', summed and PCA
- D 'Medical_Keyword_10',  summed and PCA
- D 'Medical_Keyword_11',  summed and PCA
- D 'Medical_Keyword_12',  summed and PCA
- D 'Medical_Keyword_13',  summed and PCA
- D 'Medical_Keyword_14',  summed and PCA
- D 'Medical_Keyword_15',  summed and PCA
- D 'Medical_Keyword_16',  summed and PCA
- D 'Medical_Keyword_17',  summed and PCA
- D 'Medical_Keyword_18',  summed and PCA
- D 'Medical_Keyword_19',  summed and PCA
- D 'Medical_Keyword_20',  summed and PCA
- D 'Medical_Keyword_21',  summed and PCA
- D 'Medical_Keyword_22',  summed and PCA
- D 'Medical_Keyword_23',  summed and PCA
- D 'Medical_Keyword_24',  summed and PCA
- D 'Medical_Keyword_25',  summed and PCA
- D 'Medical_Keyword_26',  summed and PCA
- D 'Medical_Keyword_27',  summed and PCA
- D 'Medical_Keyword_28',  summed and PCA
- D 'Medical_Keyword_29',  summed and PCA
- D 'Medical_Keyword_30',  summed and PCA
- D 'Medical_Keyword_31',  summed and PCA
- D 'Medical_Keyword_32',  summed and PCA
- D 'Medical_Keyword_33',  summed and PCA
- D 'Medical_Keyword_34',  summed and PCA
- D 'Medical_Keyword_35',  summed and PCA
- D 'Medical_Keyword_36',  summed and PCA
- D 'Medical_Keyword_37',  summed and PCA
- D 'Medical_Keyword_38',  summed and PCA
- D 'Medical_Keyword_39',  summed and PCA
- D 'Medical_Keyword_40',  summed and PCA
- D 'Medical_Keyword_41',  summed and PCA
- D 'Medical_Keyword_42',  summed and PCA
- D 'Medical_Keyword_43',  summed and PCA
- D 'Medical_Keyword_44',  summed and PCA
- D 'Medical_Keyword_45',  summed and PCA
- D 'Medical_Keyword_46',  summed and PCA
- D 'Medical_Keyword_47',  summed and PCA
- D 'Medical_Keyword_48',  summed and PCA


- nrounds is driving up the tuning times
- should we use early stopping for the xgb classifier?