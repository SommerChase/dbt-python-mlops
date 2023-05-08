# This model is to show people that I'm not crazy and dbt-python is cool.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import make_column_transformer

def model(dbt, session):

    dbt.config(
        materialized="table",
        packages=["pandas", "scikit-learn"]
        )

    p = dbt.ref("policies").to_pandas()
    p.columns = p.columns.str.lower()

    # Essentially a select statement:
    p_trimmed = p[[
                      "state_abbreviation"
                    , "risk_1"
                    , "segment_1"
                    , "risk_2"
                    , "prior_insurance"
                    , "prior_insurance_2"
                    , "bi_limit"
                    , "credit_score"
                    , "home_owner"
                    , "premium"
                    ]]

    p_continuous = p_trimmed.select_dtypes(include=[float])
    p_continuous.fillna(0, inplace=True)

    # Select all categorical columns to be encoded
    p_categorical = p_trimmed.select_dtypes(include=[object])

    #https://towardsdatascience.com/scikit-learn-1-1-comes-with-an-improved-onehotencoder-5a1f939da190
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(p_categorical)
    onehotlablels = ohe.transform(p_categorical)
    encoded_p = pd.DataFrame(
        onehotlablels,
        columns=ohe.get_feature_names_out()
    )

    final_transformed_data = encoded_p.merge(p_continuous,
                                           how="inner",
                                           left_index=True,
                                           right_index=True)
    
    # Select all columns except last (output column):
    X = final_transformed_data.iloc[:, 0:-1]
    y = final_transformed_data["initial_full_term_written_premium"]




    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=1337
                                                    )

    # Fit dem models bruh
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Put them to the test:
    lin_reg_preds = lin_reg.predict(X_test)

    X_train["x_train"] = 1
    X_train_merge = X_train.iloc[:, -1]

    X_test["pred_lin_reg"] = lin_reg.predict(X_test)
    X_test["x_test"] = 1
    X_test_merge = X_test.loc[:, ["x_test", "pred_lin_reg"]]


    y_train = pd.DataFrame(y_train)
    y_train["y_train"] = 1
    y_train_merge = y_train.iloc[:, -1]

    y_test = pd.DataFrame(y_test)
    y_test["y_test"] = 1
    y_test_merge = y_test.iloc[:, -1]

    final_transformed_data["is_x_train"] = X_train_merge
    final_transformed_data["is_x_test"] = X_test_merge["x_test"]
    final_transformed_data["is_y_train"] = y_train_merge
    final_transformed_data["is_y_test"] = y_test_merge
    final_transformed_data["pred_lin_reg"] = X_test_merge["pred_lin_reg"]

    return final_transformed_data
