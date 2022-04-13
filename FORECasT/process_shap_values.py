import pickle as pkl
import numpy as np
import shap

# Load shap values FORECasT/shap_save_data/shapley_values/Oligo_58_D3_L-4C5R5_global_shap_values_1.pkl
with open('FORECasT/shap_save_data/shapley_values/global_explanations/D3/n_1000/nsamples=1e05/Oligo_58_D3_L-4C5R5_global_shap_values_3.pkl',
          'rb') as f:
    shap_values = np.array(pkl.load(f))
    # set all the nan values in the shap_values array to 0.0
    shap_values[np.isnan(shap_values)] = 0.0

# Load FORECasT/explanation_datasets/Oligo_58_D3_L-4C5R5_1.pkl
with open('FORECasT/explanation_datasets/dataset_size_1000/Oligo_58_D3_L-4C5R5.pkl', 'rb') as f:
    features = pkl.load(f)

shap.summary_plot(shap_values, features)
