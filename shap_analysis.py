import shap
import xgboost as xgb
import pandas as pd
import subprocess
import os
import csv
import sys
import io

'''
In the following file, we will use the SHAP package to analyze an example XGBoost model. Consequently, we attempt to 
extent the shap explanation to the FORECasT model.

Feature explanations Boston housing example:
https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html 
'''


def boston_housing_example():
    X, y = shap.datasets.boston()
    print('Input data:')
    print(X.head())
    print('Output data (house price in $1.000):')
    print(y)

    model = xgb.XGBRegressor().fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    print('shap values:')
    print(shap_values)

    shap.plots.scatter(shap_values[:, "AGE"], color=shap_values)


def read_oligos(OLIGO_FILE):
    oligo_data = pd.read_pickle(OLIGO_FILE)

    return oligo_data


if __name__ == '__main__':
    df = read_oligos("FORECasT/train/Tijsterman_Analyser/Oligo_40")
    print(df.columns)