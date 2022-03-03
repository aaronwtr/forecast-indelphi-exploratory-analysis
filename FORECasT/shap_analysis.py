import pickle as pkl

# open pickle file as dataframe
with open('shap_values/D10_num_background_data_10.pkl', 'rb') as f:
    shap_values = pkl.load(f)

print(shap_values)
