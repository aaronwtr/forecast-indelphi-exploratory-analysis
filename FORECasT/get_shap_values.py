import pandas as pd
import numpy as np
import shap
from shap import KernelExplainer
import os
import matplotlib.pyplot as plt
from warnings import simplefilter
from tqdm import tqdm
import pickle
import io
import csv

'''
In this script, the FORECasT pre-trained model is implemented and SHAP analysis is consequently performed on top of this
implementation.
'''


def readTheta(theta_file):
    f = io.open(theta_file)
    train_set = f.readline()[:-1].split(',')
    feature_columns, theta = [], []
    for toks in csv.reader(f, delimiter='\t'):
        feature_columns.append(toks[0])
        theta.append(eval(toks[1]))
    return theta, train_set, feature_columns


def getKernelExplainerModelInput(instances, current_oligo, **kwargs):
    """
    Get the input data in the correct format for the KernelExplainer. Note that for the final data, please set a
    data_size.
    """
    shap_input = instances.iloc[:, 0:instances.shape[1] - 3]

    indel_idx = list(instances[:]['Indel'])

    indices = ['Oligo_' + str(current_oligo) + '_' + j for j in indel_idx]

    shap_input.index = indices

    shap_input = shap_input.fillna(0.0)

    return shap_input


def reshapeModelOutput(repair_outcome):
    """'
    Reshape the model output so that the prediction model outputs a vector of probabilities for each sample corresponding
    with the input dimensions in the input data.
    """

    model_output = []

    for value in repair_outcome.values():
        model_output.append(value)

    return np.array(model_output)


def model(x):
    feature_columns = list(background_df.columns)
    return predictionModel(x, DEFAULT_MODEL, feature_columns)


def predictionModel(input_data, pre_trained_model, feature_columns, plot=False):
    theta, train_set, theta_feature_columns = readTheta(pre_trained_model)
    theta_dict = {k: v for k, v in zip(theta_feature_columns, theta)}
    theta_feature_dict = {}
    for feature in feature_columns:
        if feature in theta_dict:
            theta_feature_dict[feature] = theta_dict[feature]

    theta = list(theta_feature_dict.values())

    preds = []
    for i in range(len(input_data)):
        preds.append(np.exp(sum(theta * input_data[i])))

    sum_preds = np.sum(preds)
    repair_outcome_freqs_profile = np.array([pred / (1 + sum_preds) for pred in preds])

    return repair_outcome_freqs_profile


def getBackgroundData(guidedata, ioi):
    """
    Get the background data for the SHAP analysis.
    """
    indel_idx = ioi.split('_')[2]
    oligo_of_interest = int(ioi.split('_')[1])

    oligo_idx = 0
    oligo_data = 0
    num_samples = 200

    current_oligo = int(guidedata['ID'][oligo_idx][5:])
    while current_oligo != oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guidedata['ID'][oligo_idx][5:])

    samples = pd.read_pickle(
        f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    )
    proc_samples = getKernelExplainerModelInput(samples, current_oligo)

    oligo_idx = 0
    print('Collecting background data...')
    background_data = pd.DataFrame(columns=proc_samples.columns)
    pbar = tqdm(total=num_samples)
    while oligo_data < num_samples:
        current_oligo = guidedata['ID'][oligo_idx][5:]
        oligo_name = str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in tijsterman_oligos:
            oligo_idx += 1
            continue

        feature_data = pd.read_pickle(
            f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(
                current_oligo)
        )

        model_df_temp = getKernelExplainerModelInput(feature_data, current_oligo)
        for index in model_df_temp.index:
            df_indel_idx = index.split('_')[2]
            if index == ioi:
                continue

            if df_indel_idx == indel_idx:
                background_data.loc[index] = model_df_temp.loc[index]
                oligo_data += 1
                pbar.update(1)
                break

        oligo_idx += 1

    pbar.close()

    return background_data


def getBackgroundDataZeros(guidedata, ioi):
    """
    Get the background data for the SHAP analysis.
    """
    indel_idx = ioi.split('_')[2]
    oligo_of_interest = int(ioi.split('_')[1])

    oligo_idx = 0
    oligo_data = 0
    num_samples = 200

    current_oligo = int(guidedata['ID'][oligo_idx][5:])
    while current_oligo != oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guidedata['ID'][oligo_idx][5:])

    samples = pd.read_pickle(
        f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    )
    proc_samples = getKernelExplainerModelInput(samples, current_oligo)

    background_data = pd.DataFrame(columns=proc_samples.columns)
    background_data.loc[0, :] = np.zeros(len(proc_samples.columns))

    return background_data


def getExplanationData(guidedata, ioi):
    indel_idx = ioi.split('_')[2]
    oligo_of_interest = int(ioi.split('_')[1])

    oligo_idx = 0
    oligo_data = 0
    num_samples = 200

    current_oligo = int(guidedata['ID'][oligo_idx][5:])
    while current_oligo != oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guidedata['ID'][oligo_idx][5:])

    samples = pd.read_pickle(
        f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    )

    proc_samples = getKernelExplainerModelInput(samples, current_oligo)
    backgroundIndels = list(background_df.index)

    print('Collecting explanation data...')
    explanation_data = pd.DataFrame(columns=proc_samples.columns)
    pbar = tqdm(total=num_samples)
    while oligo_data < num_samples:
        current_oligo = guidedata['ID'][oligo_idx][5:]
        oligo_name = str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in tijsterman_oligos:
            oligo_idx += 1
            continue

        feature_data = pd.read_pickle(
            f"{FORECasT_path}/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(
                current_oligo)
        )

        model_df_temp = getKernelExplainerModelInput(feature_data, current_oligo)
        for index in model_df_temp.index:
            df_indel_idx = index.split('_')[2]
            if index in backgroundIndels:
                continue

            if df_indel_idx == indel_idx:
                explanation_data.loc[index] = model_df_temp.loc[index]
                oligo_data += 1
                pbar.update(1)
                break

        oligo_idx += 1

    pbar.close()

    return explanation_data


def plotShapleyValues():
    """
    Plot the shapley value for the given input. If we are explaining a single prediction, we will plot a force plot for
    that single prediction. If we are explaining a set of predictions, we will plot a summary beeswarm plot for the
    entire explanation dataset.
    """

    shap.initjs()
    plt.rcParams['ytick.labelsize'] = 'small'

    if explain_sample == 'one':
        force = shap.force_plot(
            expected_value, shap_values, explanation_df.iloc[explain_prediction, :], list(explanation_df.columns),
            show=False, contribution_threshold=0.1, text_rotation=0.4
        )
        shap.save_html(
            f'{FORECasT_path}/shap_values_visualizations/force_plots/{list(explanation_df.index)[explain_prediction]}_forceplot.html',
            force
        )
        plt.clf()
        plt.close()

    if explain_sample == 'all':
        shap.summary_plot(shap_values, background_df, list(background_df.columns))


def getShapleyValues(model, background_data, explanation_data, explain_sample='all', link='logit'):
    """
    Compute the SHAP values for the explanation data. If no specific sample is specified, the SHAP values of the entire
    explanation set are computed. If explain_sample is one, then automatically the first instance of the explanation set
    is explained.

    A copy of the Shapley values array is saved to shap_save_data/shapley_values. This is a tuple with index 0 being the
    Shapley value array and index 1 being the expected value for the explanation set. This expected value is needed if
    we want to generate local SHAP plots.
    :return: Returns either a Shapley value matrix, or a tuple with the Shapley value matrix and the expected value.
    """

    explainer = KernelExplainer(model, background_data, link=link)

    if explain_sample == 'all':
        if os.path.isfile(f'{FORECasT_path}/shap_save_data/shapley_values/{indel_of_interest}_global_shap_values.pkl'):
            shapley_val = pickle.load(
                open(f'{FORECasT_path}/shap_save_data/shapley_values/{indel_of_interest}_global_shap_values.pkl', 'rb'))
        else:
            shapley_val = explainer.shap_values(explanation_data, nsamples=10**5)
            with open(f'{FORECasT_path}/shap_save_data/shapley_values/{indel_of_interest}_global_shap_values.pkl',
                      'wb') as file:
                pickle.dump(shapley_val, file)
            file.close()

        return shapley_val

    elif explain_sample == 'one':
        if os.path.isfile(f'{FORECasT_path}/shap_save_data/shapley_values/{indel_of_interest}_local_shap_values.pkl'):
            shapley_val, expected_val = pickle.load(
                open(f'{FORECasT_path}/shap_save_data/shapley_values/{indel_of_interest}_local_shap_values.pkl', 'rb'))
        else:
            shapley_val = explainer.shap_values(explanation_data.iloc[0, :], nsamples="auto")
            expected_val = explainer.expected_value
            with open(f'{FORECasT_path}/shap_save_data/shapley_values/{indel_of_interest}_local_shap_values.pkl',
                      'wb') as file:
                pickle.dump((shap, expected_val), file)
            file.close()

        return shapley_val, expected_val


if __name__ == '__main__':
    # Note that not all oligo's in the guideset are present in the Tijsterman data present locally.
    # Get the absolute path to where the folder that contains the FORECasT code is located.
    FORECasT_path_tmp = os.path.dirname(os.path.abspath(__file__))
    FORECasT_path = FORECasT_path_tmp.replace(os.sep, '/')
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv(f"{FORECasT_path}/guideset_data.txt", sep='\t')
    tijsterman_oligos = os.listdir(f"{FORECasT_path}/train/Tijsterman_Analyser")
    DEFAULT_MODEL = f"{FORECasT_path}/tmp_model_thetas.txt"
    indel_of_interest = "Oligo_58_D3_L-4C5R5"

    # if os.path.isfile(f"{FORECasT_path}/background_datasets/{indel_of_interest}.pkl"):
    #     background_df = pd.read_pickle(f"{FORECasT_path}/background_datasets/{indel_of_interest}.pkl")
    # else:
    #     background_df = getBackgroundData(guideset, indel_of_interest)
    #     background_df.to_pickle(f"{FORECasT_path}/background_datasets/{indel_of_interest}.pkl")

    background_df = getBackgroundDataZeros(guideset, indel_of_interest)
    if os.path.isfile(f"{FORECasT_path}/explanation_datasets/dataset_size_1000/{indel_of_interest}.pkl"):
        explanation_df = pd.read_pickle(f"{FORECasT_path}/explanation_datasets/dataset_size_1000/{indel_of_interest}.pkl")
    else:
        explanation_df = getExplanationData(guideset, indel_of_interest)
        explanation_df.to_pickle(f"{FORECasT_path}/explanation_datasets/{indel_of_interest}.pkl")

    explain_prediction = 0  # note that the repair outcome of interest is in the first row of the explanation data
    explain_sample = 'all'

    if explain_sample == 'all':
        print("Getting Shapley values for all samples...")
        shap_values = getShapleyValues(model, background_df, explanation_df, explain_sample=explain_sample)
    else:
        print("Getting Shapley values for one sample...")
        shap_values, expected_value = getShapleyValues(model, background_df, explanation_df,
                                                       explain_sample=explain_sample)

    # TODO 1: Increase size of explanation dataset.
    # TODO 2: Increase nsamples by 1 order of magnitude.
    # TODO 3: Put all the hyperparameters together, maybe in a config file?
