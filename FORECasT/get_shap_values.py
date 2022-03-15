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


def getKernelExplainerModelInput(instances, current_oligo, **kwargs):
    """
    Get the input data in the correct format for the KernelExplainer. Note that for the final data, please set a
    data_size.
    """
    shap_input = instances.iloc[:, 0:instances.shape[1] - 3]

    indel_idx = list(instances[:]['Indel'])

    indices = ['Oligo_' + str(current_oligo) + '_' + j for j in indel_idx]

    shap_input.index = indices

    # Note that n will be the number of samples per oligo
    # if 'data_size' in kwargs:
    #     n = kwargs['data_size']
    #     shap_input = shap_input.sample(n)

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


def getBackgroundData(guidedata, ioi):
    """
    Get the background data for the SHAP analysis.
    """
    indel_idx = ioi.split('_')[2]
    oligo_of_interest = int(ioi.split('_')[1])

    oligo_idx = 0
    oligo_data = 0
    num_samples = 100

    current_oligo = int(guidedata['ID'][oligo_idx][5:])
    while current_oligo is not oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guidedata['ID'][oligo_idx][5:])

    samples = pd.read_pickle(
        "FORECasT/train/Tijsterman_Analyser/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' + str(current_oligo))
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
            "FORECasT/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo))

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


def getExplanationData(guidedata):
    indel_of_interest = "Oligo_58_D3_L-4C5R5"
    indel_idx = indel_of_interest.split('_')[2]
    oligo_of_interest = indel_of_interest.split('_')[1]
    df_container = []

    oligo_idx = 0
    oligo_data = 0
    num_samples = 100

    current_oligo = int(guidedata['ID'][oligo_idx][5:])
    while current_oligo is not oligo_of_interest:
        oligo_idx += 1
        current_oligo = int(guidedata['ID'][oligo_idx][5:])

    proc_samples = getKernelExplainerModelInput(samples, current_oligo)
    repair_outcome_of_interest = proc_samples.loc[indel_of_interest]  # note that this is a df series

    return


def getSHAPValue(model, background_data, explanation_data, explain_sample='all', link='logit'):
    """
    Compute the SHAP values for the explanation data. If no specific sample is specified, the SHAP values of the entire
    explanation set are computed. If explain_sample is one, then automatically the first instance of the explanation set
    is explained.
    """

    num_samples = 10

    explainer = KernelExplainer(model, background_data, link=link)

    if explain_sample == 'all':
        shap = explainer.shap_values(explanation_data, nsamples=num_samples)

    elif explain_sample == 'one':
        shap = explainer.shap_values(explanation_data.iloc[0, :], nsamples=num_samples)
    else:
        shap = explainer.shap_values(explanation_data.iloc[int(explain_sample), :], nsamples=num_samples)

    return shap, explainer


if __name__ == '__main__':
    # Note that not all oligo's in the guideset are present in the Tijsterman data present locally.
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv("FORECasT/guideset_data.txt", sep='\t')
    tijsterman_oligos = os.listdir("FORECasT/train/Tijsterman_Analyser")

    target_seq_list = []
    pam_idx_list = []
    oligos_list = []
    indels = []
    pred_data = []

    indel_of_interest = "Oligo_58_D3_L-4C5R5"
    model_df = getBackgroundData(guideset, indel_of_interest)
    model_input = model_df.to_numpy()  # SHAP expects a numpy array

    print(model_df)

    # oligo_data = 0
    # dfs_container_ex = []
    # explanation_data_size = 1000
    # sample_size_explanation = int(np.round(explanation_data_size / num_oligos))
    #
    # print('\nCollecting explanation dataset...')
    # # Get the explanation data set
    # while oligo_data < num_oligos:
    #     current_oligo = guideset['ID'][oligo_idx][5:]
    #     oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
    #     if oligo_name not in tijsterman_oligos:
    #         oligo_idx += 1
    #         continue
    #
    #     feature_data = pd.read_pickle(
    #         "FORECasT/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo))
    #
    #     model_df_temp = getKernelExplainerModelInput(feature_data, current_oligo, sample_size_explanation)
    #     for i in range(len(list(model_df_temp.index))):
    #         oligos_list.append(current_oligo)
    #
    #     dfs_container_ex.append(model_df_temp)
    #     oligo_data += 1
    #     oligo_idx += 1

    explain_prediction = 0
    explanation_data_tmp = pd.concat(dfs_container_ex)
    print('Explanation dataset: \n', explanation_data_tmp)
    explanation_data = explanation_data_tmp.to_numpy()
    explain_sample = 'all'

    # shap_values, ex = getSHAPValue(model, model_df, explanation_data, explain_sample='all')

    # shap.initjs()
    # plt.rcParams['ytick.labelsize'] = 'small'
    #
    # if explain_sample == 'one':
    #     force = shap.force_plot(ex.expected_value, shap_values, model_df.iloc[explain_prediction, :], show=False)
    #     shap.save_html(f'FORECasT/shap_values/force_plots/forceplot_{list(model_df.index)[explain_prediction]}.html',
    #                    force)
    #     plt.clf()
    #     plt.close()
    #
    # if explain_sample == 'all':
    #     shap.summary_plot(shap_values, model_df, list(model_df.columns))
    #
    # # save shap_values to .pkl file
    # with open('FORECasT/shap_values/shapley_values/' + str(
    #         list(model_df.index)[explain_prediction]) + '_' + 'num_background_data' + '_' +
    #           str(num_oligos) + '.pkl', 'wb') as f:
    #     pkl.dump(shap_values, f)
    #     f.close()

    # TODO: Make the force plots for the repair outcomes with the highest probability of occuring.

    # TODO: Figure out what the features mean (check FORECasT github).

    # TODO: Compare logistic regression feature weigths to the shapley values. Note that the logistic regression feature
    # TODO: weights are global whereas shap values are local explanations.
