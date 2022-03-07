import pandas as pd
import numpy as np
import shap
from shap import KernelExplainer, Explainer
import os
import matplotlib.pyplot as plt
import pickle as pkl
from warnings import simplefilter
from tqdm import tqdm

from predictor.model import readTheta, computePredictedProfile
from predictor.predict import DEFAULT_MODEL, predictMutations, INDELGENTARGET_EXE, fetchRepReads

'''
In this script, the FORECasT pre-trained model is implemented and SHAP analysis is consequently performed on top of this
implementation.
'''


def getKernelExplainerModelInput(instances, current_oligo):
    model_input = instances.iloc[:, 0:instances.shape[1] - 3]

    indel_idx = list(instances[:]['Indel'])

    indices = ['Oligo_' + str(current_oligo) + '_' + j for j in indel_idx]

    model_input.index = indices

    # Note that n will be the number of samples per oligo
    model_input = model_input.sample(n=sampling_num)

    return model_input


def reshapeModelOutput(repair_outcome):
    ''''
    Reshape the model output so that the prediction model outputs a vector of probabilities for each sample corresponding
    with the input dimensions in the input data.
    '''

    model_output = []

    for value in repair_outcome.values():
        model_output.append(value)

    return np.array(model_output)


def model(x):
    feature_columns = list(model_df.columns)
    return predictionModel(x, DEFAULT_MODEL, feature_columns)


def predictionModel(input_data, pre_trained_model, feature_columns, plot=False):
    print("Performing predictions for input samples...\n")

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


def getSHAPValue(model, background_data, explanation_data, explain_sample='all', link='logit'):
    '''
    Compute the SHAP values for the explanation data. If no specific sample is specified, the SHAP values of the entire
    explanation set are computed. If explain_sample is one, then automatically the first instance of the explanation set
    is explained.
    '''

    num_samples = 10

    explainer = KernelExplainer(model, background_data, link=link)

    if explain_sample == 'all':
        shap = explainer.shap_values(background_data, nsamples=num_samples)

    elif explain_sample == 'one':
        shap = explainer.shap_values(background_data.iloc[0, :], nsamples=num_samples)
    else:
        shap = explainer.shap_values(background_data.iloc[int(explain_sample), :], nsamples=num_samples)

    return shap, explainer


if __name__ == '__main__':
    # Note that not all oligo's in the guideset are present in the Tijsterman data present locally.
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv("FORECasT/guideset_data.txt", sep='\t')
    tijsterman_oligos = os.listdir('FORECasT/train/Tijsterman_Analyser')

    dfs_container = []
    oligo_data = 0
    oligo_idx = 0

    target_seq_list = []
    pam_idx_list = []
    oligos_list = []
    indels = []
    pred_data = []

    oligo_idx = 0
    num_oligos = 1
    size_background_data = 100
    sampling_num = int(size_background_data / num_oligos)

    print('Collecting background data...')

    # Get the background data
    while oligo_data < num_oligos:
        current_oligo = guideset['ID'][oligo_idx][5:]
        oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in tijsterman_oligos:
            oligo_idx += 1
            continue

        feature_data = pd.read_pickle(
            "FORECasT/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo))

        model_df_temp = getKernelExplainerModelInput(feature_data, current_oligo)
        for i in range(len(list(model_df_temp.index))):
            oligos_list.append(current_oligo)

        dfs_container.append(model_df_temp)
        oligo_data += 1
        oligo_idx += 1

    model_df = pd.concat(dfs_container)
    model_input = model_df.to_numpy()  # SHAP expects ndarray

    oligo_data = 0
    dfs_container_ex = []
    print('\nCollecting explanation dataset...')
    # Get the explanation data set
    while oligo_data < num_oligos:
        current_oligo = guideset['ID'][oligo_idx][5:]
        oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in tijsterman_oligos:
            oligo_idx += 1
            continue

        feature_data = pd.read_pickle(
            "FORECasT/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo))

        model_df_temp = getKernelExplainerModelInput(feature_data, current_oligo)
        for i in range(len(list(model_df_temp.index))):
            oligos_list.append(current_oligo)

        dfs_container_ex.append(model_df_temp)
        oligo_data += 1
        oligo_idx += 1

    explain_prediction = 0
    explanation_data_tmp = pd.concat(dfs_container_ex)
    explanation_data = explanation_data_tmp.to_numpy()
    explain_sample = 'all'

    shap_values, ex = getSHAPValue(model, model_df, explanation_data, explain_sample='all')
    print(type(shap_values))

    shap.initjs()
    plt.rcParams['ytick.labelsize'] = 'small'

    if explain_sample == 'one':
        force = shap.force_plot(ex.expected_value, shap_values, model_df.iloc[explain_prediction, :], show=False)
        shap.save_html(f'FORECasT/shap_values/force_plots/forceplot_{list(model_df.index)[explain_prediction]}.html',
                       force)
        plt.clf()
        plt.close()

    if explain_sample == 'all':
        shap.summary_plot(shap_values, model_df, list(model_df.columns))

    # save shap_values to .pkl file
    with open('FORECasT/shap_values/shapley_values/' + str(
            list(model_df.index)[explain_prediction]) + '_' + 'num_background_data' + '_' +
              str(num_oligos) + '.pkl', 'wb') as f:
        pkl.dump(shap_values, f)
        f.close()

    # TODO: Make the force plots for the repair outcomes with the highest probability of occuring.

    # TODO: Figure out what the features mean (check FORECasT github).
    
    # TODO: Figure out how to scale up the summary plot.

    # TODO: Compare logistic regression feature weigths to the shapley values.
