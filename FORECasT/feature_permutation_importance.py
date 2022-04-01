from sklearn.inspection import permutation_importance
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from get_shap_values import model


def formatModelInput(instances, current_oligo, **kwargs):
    """
    Get the input data in the correct format for the KernelExplainer. Note that for the final data, please set a
    data_size.
    """
    output = instances['Frac Sample Reads']
    shap_input = instances.iloc[:, 0:instances.shape[1] - 3]

    indel_idx = list(instances[:]['Indel'])

    indices = ['Oligo_' + str(current_oligo) + '_' + j for j in indel_idx]

    shap_input.index = indices
    output.index = indices

    shap_input = shap_input.fillna(0.0)

    return shap_input, output


def getData(guidedata, ioi):
    indel_idx = ioi.split('_')[2]
    oligo_of_interest = int(ioi.split('_')[1])

    oligo_idx = 0
    oligo_data = 0
    num_samples = 100

    current_oligo = int(guidedata['ID'][oligo_idx][5:])
    while current_oligo != oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guidedata['ID'][oligo_idx][5:])

    samples = pd.read_pickle(
        f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    )

    proc_samples, y = formatModelInput(samples, current_oligo)
    backgroundIndels = list(background_df.index)

    print('Collecting explanation data...')
    data = pd.DataFrame(columns=proc_samples.columns)
    y = pd.DataFrame(columns=['Frac Sample Reads'])
    pbar = tqdm(total=num_samples)
    while oligo_data < num_samples:
        current_oligo = guidedata['ID'][oligo_idx][5:]
        oligo_name = str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in tijsterman_oligos:
            oligo_idx += 1
            continue

        feature_data = pd.read_pickle(
            f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        )

        model_df_temp, y_temp = formatModelInput(feature_data, current_oligo)
        for index in model_df_temp.index:
            df_indel_idx = index.split('_')[2]
            if index in backgroundIndels:
                continue

            if df_indel_idx == indel_idx:
                data.loc[index] = model_df_temp.loc[index]
                y.loc[index] = y_temp.loc[index]
                oligo_data += 1
                pbar.update(1)
                break

        oligo_idx += 1

    print(len(data.index))
    print(len(y.index))
    pbar.close()

    return data, y


def KL_divergence(p, q):
    """
    Calculates the symmetric KL-divergence between two probability distributions.
    """
    return np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))


def feature_permutation_importance(f, X, y, n_repeats=30, random_state=0):
    """
    Performs a feature permutation importance analysis on the specified model, here referred to as f. The model must be 
    a Scikit-Learn model that supports the .fit() and .predict() methods. The X and y arguments must be the X and y 
    dataframes from the get_shap_values.py getBackgroundData function. It is preferred that these are test data samples,
    because this makes it possible to highlight which features contribute the most to the generalization power of the 
    inspected model. 
    """
    return permutation_importance(f, X, y, n_repeats=n_repeats, random_state=random_state)


if __name__ == '__main__':
    FORECasT_path_tmp = os.path.dirname(os.path.abspath(__file__))
    FORECasT_path = FORECasT_path_tmp.replace(os.sep, '/')
    guideset = pd.read_csv(f"{FORECasT_path}/guideset_data.txt", sep='\t')
    tijsterman_oligos = os.listdir(f"{FORECasT_path}/train/Tijsterman_Analyser")
    ioi = "Oligo_58_D3_L-4C5R5"
    indel_idx = ioi.split('_')[2]
    oligo_of_interest = int(ioi.split('_')[1])
    background_df = pd.read_pickle(f"{FORECasT_path}/background_datasets/dataset_size_100/{ioi}.pkl")

    oligo_idx = 0
    oligo_data = 0
    num_samples = 1000

    current_oligo = int(guideset['ID'][oligo_idx][5:])
    while current_oligo != oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guideset['ID'][oligo_idx][5:])

    samples = pd.read_pickle(
        f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    )

    if os.path.isfile(f"{FORECasT_path}/feature_importance_datasets/{ioi}.pkl"):
        X = pd.read_pickle(f"{FORECasT_path}/feature_importance_datasets/{ioi}.pkl")
        y = pd.read_pickle(f"{FORECasT_path}/feature_importance_datasets/{ioi}_y.pkl")
    else:
        X, y = getData(guideset, ioi)
        X.to_pickle(f"{FORECasT_path}/feature_importance_datasets/{ioi}.pkl")
        y.to_pickle(f"{FORECasT_path}/feature_importance_datasets/{ioi}_y.pkl")

    print(X)
    print(y)

    # imp_means, imp_stds, feat_imps = feature_permutation_importance(
