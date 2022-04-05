import pickle as pkl
import numpy as np
import shap

# Load shap values FORECasT/shap_save_data/shapley_values/Oligo_58_D3_L-4C5R5_global_shap_values_1.pkl
with open('FORECasT/shap_save_data/shapley_values/Oligo_58_D3_L-4C5R5_global_shap_values.pkl', 'rb') as f:
    shap_values = np.array(pkl.load(f))

# Load FORECasT/explanation_datasets/Oligo_58_D3_L-4C5R5_1.pkl
with open('FORECasT/explanation_datasets/dataset_size_1000/Oligo_58_D3_L-4C5R5.pkl', 'rb') as f:
    features = pkl.load(f)

shap.summary_plot(shap_values, features)
