import pandas as pd
import numpy as np
import shap
from shap import KernelExplainer
import subprocess
import os
import io
import random
import matplotlib.pyplot as plt
import pickle as pkl
from warnings import simplefilter
from tqdm import tqdm

from predictor.features import calculateFeaturesForGenIndelFile, readFeaturesData
from predictor.model import readTheta, computePredictedProfile
from predictor.predict import DEFAULT_MODEL, predictMutations, INDELGENTARGET_EXE, fetchRepReads
from selftarget.indel import tokFullIndel
from selftarget.profile import fetchIndelSizeCounts
from selftarget.view import plotProfiles, getAvgPreds

'''
In this script, the FORECasT pre-trained model is implemented and SHAP analysis is consequently performed on top of this
implementation.
'''


# def getProfileCounts(profile):
#     total = sum([profile[x] for x in profile])
#     if total == 0:
#         return []
#     indel_total = total
#     if '-' in profile:
#         indel_total -= profile['-']
#         null_perc = profile['-'] * 100.0 / indel_total if indel_total != 0 else 100.0
#         null_profile = (profile['-'], '-', profile['-'] * 100.0 / total, null_perc)
#     counts = [(profile[x], x, profile[x] * 100.0 / total, profile[x] * 100.0 / indel_total) for x in profile if
#               x != '-']
#     counts.sort(reverse=True)
#     if '-' in profile:
#         counts = [null_profile] + counts
#     return counts
#
#
# def writePredictedProfileToSummary(p1, fout):
#     counts = getProfileCounts(p1)
#     for cnt, indel, _, _ in counts:
#         if cnt < 0.5: break
#         fout.write(u'%s\t-\t%d\n' % (indel, np.round(cnt)))
#
#
# def writePredictedRepReadsToFile(p1, rep_reads, fout):
#     counts = getProfileCounts(p1)
#     idx = 0
#     for cnt, indel, _, _ in counts:
#         if cnt < 0.5: break
#         fout.write(u'%d\t%s\t%s\n' % (idx, rep_reads[indel], indel))
#         idx += 1
#
#
# def writeProfilesToFile(out_prefix, profiles_and_rr, write_rr=False):
#     fout = io.open(out_prefix + '_predictedindelsummary.txt', 'w')
#     if write_rr: fout_rr = io.open(out_prefix + '_predictedreads.txt', 'w')
#     for (guide_id, prof, rep_reads, in_frame) in profiles_and_rr:
#         if len(profiles_and_rr) > 1:
#             id_str = u'@@@%s\t%.3f\n' % (guide_id, in_frame)
#             fout.write(id_str)
#             if write_rr:
#                 fout_rr.write(id_str)
#         writePredictedProfileToSummary(prof, fout)
#         if write_rr:
#             writePredictedRepReadsToFile(prof, rep_reads, fout_rr)
#     fout.close()
#
#
# def predictMutationsSingle(target_seq, pam_idx, out_prefix, theta_file=DEFAULT_MODEL):
#     print('Predicting mutations...')
#     p_predict, rep_reads, in_frame_perc = predictMutations(theta_file, target_seq, pam_idx)
#     print('Writing to file...')
#     writeProfilesToFile(out_prefix, [('Test Guide', p_predict, rep_reads, in_frame_perc)], write_rr=True)
#     print('Done!')
#
#
# def getIndels(instances):
#     return list(instances[:]['Indel'])
#
#
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

    print(repair_outcome_freqs_profile)
    # # repair_outcome_freqs = reshapeModelOutput(repair_outcome_freqs_profile)

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
    num_oligos = 10
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
    num_oligos = 10
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
    model_input = model_df.to_numpy()   # SHAP expects ndarray

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

    explain_sample = 0
    explanation_data_tmp = pd.concat(dfs_container_ex)
    explanation_data = explanation_data_tmp.to_numpy()

    shap_values, ex = getSHAPValue(model, model_df, explanation_data, explain_sample='one')
    shap.initjs()
    plot = shap.force_plot(ex.expected_value, shap_values, model_df.iloc[explain_sample, :], show=False)

    shap.save_html(f'FORECasT/shap_values/forceplot_{list(model_df.index)[explain_sample]}.html', plot)

    print(model_df)

    # TODO: Debug shap output: f(x) does not seem to match up with model output. Negative values are present which is
    # TODO: not expected as well.

    # save shap_values to .pkl file
    # with open('FORECasT/shap_values/' + str() + '_' + 'num_background_data' + '_' +
    #           str(num_oligos) + '.pkl', 'wb') as f:
    #     pkl.dump(shap_values, f)
    #     f.close()
