import shap
import xgboost as xgb

'''
In the following file, we will use the SHAP package to analyze an example XGBoost model. How do we translate this to the
FORECasT model?
'''

X, y = shap.datasets.boston()
print(X)

