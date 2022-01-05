import shap
import xgboost as xgb

'''
In the following file, we will use the SHAP package to analyze an example XGBoost model. How do we translate this to the
FORECasT model?

Feature explanations:
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
