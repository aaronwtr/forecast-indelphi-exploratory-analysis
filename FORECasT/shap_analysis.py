import pickle as pkl

# open pickle file as dataframe
with open('shap_values/shapley_values/Oligo_38_D29_L-9C2R23_num_background_data_10.pkl', 'rb') as f:
    shap_values = pkl.load(f)

