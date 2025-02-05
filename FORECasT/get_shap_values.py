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
import config

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
    return predictionModel(x, config.DEFAULT_MODEL, feature_columns)


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
    Get a background matrix for the SHAP analysis.
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
        f"E:/Aaron/Nanobiology/MSc/Year3/MEP/train/Tijsterman_Analyser/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    )
    proc_samples = getKernelExplainerModelInput(samples, current_oligo)

    oligo_idx = 0
    print('Collecting background data...')
    background_data = pd.DataFrame(columns=proc_samples.columns)
    pbar = tqdm(total=num_samples)
    while oligo_data < num_samples:
        current_oligo = guidedata['ID'][oligo_idx][5:]
        oligo_name = str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in config.tijsterman_oligos:
            oligo_idx += 1
            continue

        feature_data = pd.read_pickle(
            f"E:/Aaron/Nanobiology/MSc/Year3/MEP/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5])
            + '_' + str(current_oligo)
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


def getBackgroundDataZeros(guidedata):
    """
    Get a background data zero vector for the SHAP analysis to represent missing features.
    """
    current_oligo = 38

    samples = pd.read_pickle(
        f"E:/Aaron/Nanobiology/MSc/Year3/MEP/train/Tijsterman_Analyser/" + str(guidedata['ID'][0][0:5]) + '_' +
        str(current_oligo)
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

    current_oligo = int(guidedata['ID'][oligo_idx][5:])
    while current_oligo != oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guidedata['ID'][oligo_idx][5:])

    samples = pd.read_pickle(
        f"{config.path}/candidate_samples/test_data/dinucleotide_insertions_most_freq/" + str(
            guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    )

    proc_samples = getKernelExplainerModelInput(samples, current_oligo)
    backgroundIndels = list(background_df.index)

    print('Collecting explanation data...')
    explanation_data = pd.DataFrame(columns=proc_samples.columns)
    pbar = tqdm(total=config.dataset_size)
    while oligo_data < config.dataset_size:
        current_oligo = guidedata['ID'][oligo_idx][5:]
        oligo_name = str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in config.hd_test_tijsterman_oligos:
            oligo_idx += 1
            continue

        feature_data = pd.read_pickle(
            f"{config.hd_test_path}/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(
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

    if config.shap_type == 'local':
        force = shap.force_plot(
            expected_value, shap_values, explanation_df.iloc[0, :], list(explanation_df.columns),
            show=False, contribution_threshold=0.1, text_rotation=0.4
        )
        shap.save_html(
            f'{config.path}/shap_values_visualizations/force_plots/{list(explanation_df.index)[0]}_forceplot.html',
            force
        )
        plt.clf()
        plt.close()

    if config.shap_type == 'global':
        shap.summary_plot(shap_values, background_df, list(background_df.columns))


def getShapleyValues(model, background_data, explanation_data, explain_sample='global', link='logit'):
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

    if isinstance(config.nsamples, float):
        nsamples = int(config.nsamples)
    elif isinstance(config.nsamples, str) and config.nsamples != 'auto':
        nsamples = int(float(config.nsamples))
    elif config.nsamples == 'auto':
        nsamples = config.nsamples
    else:
        assert isinstance(config.nsamples, int), "config.nsamples must be an int, a string or a float."
        nsamples = config.nsamples

    shap_save_path_0 = f'{config.path}/shap_save_data/shapley_values/{config.shap_type}_explanations'
    indel_name_0 = config.indel_of_interest.split('_')[2]
    exact_save_location_0 = f'{indel_name_0}/n_{config.dataset_size}/nsamples={config.nsamples}'
    num_files_0 = len(list(os.listdir(f'{shap_save_path_0}/{exact_save_location_0}')))
    file_name_prefix_0 = f'{config.indel_of_interest}_{config.shap_type}_shap_values_'

    if config.shap_type == 'global':
        if num_files_0 == config.num_files_to_obtain:
            shapley_val = pickle.load(
                open(f'{shap_save_path_0}/{exact_save_location_0}/{file_name_prefix_0}{num_files_0}.pkl', 'rb'))
        else:
            shapley_val = explainer.shap_values(explanation_data, nsamples=nsamples)

        return shapley_val

    elif config.shap_type == 'local':
        if num_files_0 == config.num_files_to_obtain:
            shapley_val, expected_val = pickle.load(
                open(f'{shap_save_path_0}/{exact_save_location_0}/{file_name_prefix_0}{num_files_0}.pkl', 'rb'))
        else:
            shapley_val = explainer.shap_values(explanation_data.iloc[0, :], nsamples=nsamples)
            expected_val = explainer.expected_value

        return shapley_val, expected_val


if __name__ == '__main__':
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')

    background_df = getBackgroundDataZeros(guideset)

    explanation_dataset_path = f'{config.path}/explanation_datasets/dataset_size_{config.dataset_size}'
    explanation_dataset_name = f'{config.indel_of_interest}.pkl'

    if os.path.isfile(f'{explanation_dataset_path}/{explanation_dataset_name}'):
        explanation_df = pd.read_pickle(f'{explanation_dataset_path}/{explanation_dataset_name}')
    else:
        explanation_df = getExplanationData(guideset, config.indel_of_interest)
        explanation_df.to_pickle(f'{explanation_dataset_path}/{explanation_dataset_name}')

    shap_save_path = f'{config.path}/shap_save_data/shapley_values/{config.shap_type}_explanations'
    indel_name = config.indel_of_interest.split('_')[2]
    exact_save_location = f'{indel_name}/n_{config.dataset_size}/nsamples={config.nsamples}'

    if config.shap_type == 'global':
        print("Getting Shapley values for all samples...")
        shap_values = getShapleyValues(model, background_df, explanation_df, explain_sample=config.shap_type)
        num_files = len(list(os.listdir(f'{shap_save_path}/{exact_save_location}')))
        file_name_prefix = f'{config.indel_of_interest}_{config.shap_type}_shap_values_'
        with open(f'{shap_save_path}/{exact_save_location}/{file_name_prefix}{num_files + 1}.pkl', 'wb') as file:
            pickle.dump(shap_values, file)
        file.close()
        print(f"Shapley values saved to {shap_save_path}/{exact_save_location}/{file_name_prefix}{num_files + 1}.pkl")
    else:
        print("Getting Shapley values for one sample...")
        shap_values, expected_value = getShapleyValues(model, background_df, explanation_df,
                                                       explain_sample=config.shap_type)
        num_files = len(list(os.listdir(f'{shap_save_path}/{exact_save_location}')))
        file_name_prefix = f'{config.indel_of_interest}_{config.shap_type}_shap_values_'
        with open(f'{shap_save_path}/{exact_save_location}/{file_name_prefix}{num_files + 1}.pkl',
                  'wb') as file:
            pickle.dump((shap_values, expected_value), file)
        file.close()
        print(f"Shapley values saved to {shap_save_path}/{exact_save_location}/{file_name_prefix}{num_files + 1}.pkl")
