import pandas as pd
import numpy as np
from shap import KernelExplainer
import subprocess
import os
import io
import random
import matplotlib.pyplot as plt
import pickle as pkl
from warnings import simplefilter

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


def getProfileCounts(profile):
    total = sum([profile[x] for x in profile])
    if total == 0:
        return []
    indel_total = total
    if '-' in profile:
        indel_total -= profile['-']
        null_perc = profile['-'] * 100.0 / indel_total if indel_total != 0 else 100.0
        null_profile = (profile['-'], '-', profile['-'] * 100.0 / total, null_perc)
    counts = [(profile[x], x, profile[x] * 100.0 / total, profile[x] * 100.0 / indel_total) for x in profile if
              x != '-']
    counts.sort(reverse=True)
    if '-' in profile:
        counts = [null_profile] + counts
    return counts


def writePredictedProfileToSummary(p1, fout):
    counts = getProfileCounts(p1)
    for cnt, indel, _, _ in counts:
        if cnt < 0.5: break
        fout.write(u'%s\t-\t%d\n' % (indel, np.round(cnt)))


def writePredictedRepReadsToFile(p1, rep_reads, fout):
    counts = getProfileCounts(p1)
    idx = 0
    for cnt, indel, _, _ in counts:
        if cnt < 0.5: break
        fout.write(u'%d\t%s\t%s\n' % (idx, rep_reads[indel], indel))
        idx += 1


def writeProfilesToFile(out_prefix, profiles_and_rr, write_rr=False):
    fout = io.open(out_prefix + '_predictedindelsummary.txt', 'w')
    if write_rr: fout_rr = io.open(out_prefix + '_predictedreads.txt', 'w')
    for (guide_id, prof, rep_reads, in_frame) in profiles_and_rr:
        if len(profiles_and_rr) > 1:
            id_str = u'@@@%s\t%.3f\n' % (guide_id, in_frame)
            fout.write(id_str)
            if write_rr:
                fout_rr.write(id_str)
        writePredictedProfileToSummary(prof, fout)
        if write_rr:
            writePredictedRepReadsToFile(prof, rep_reads, fout_rr)
    fout.close()


def predictMutationsSingle(target_seq, pam_idx, out_prefix, theta_file=DEFAULT_MODEL):
    print('Predicting mutations...')
    p_predict, rep_reads, in_frame_perc = predictMutations(theta_file, target_seq, pam_idx)
    print('Writing to file...')
    writeProfilesToFile(out_prefix, [('Test Guide', p_predict, rep_reads, in_frame_perc)], write_rr=True)
    print('Done!')


def getKernelExplainerModelInput(instances, current_oligo):
    model_input = instances.iloc[:, 0:instances.shape[1] - 3]

    indel_idx = list(instances[:]['Indel'])
    indices = ['Oligo_' + str(current_oligo) + '_' + i for i in indel_idx]

    model_input.index = indices

    return model_input


def predictMutations(input_data, theta_file, target_seq, pam_idx, add_null=True):
    theta, train_set, theta_feature_columns = readTheta(theta_file)

    # generate indels
    left_trim = 0
    tmp_genindels_file = 'tmp_genindels_%s_%d.txt' % (target_seq, random.randint(0, 100000))
    cmd = INDELGENTARGET_EXE + ' %s %d %s' % (target_seq, pam_idx, tmp_genindels_file)
    subprocess.check_call(cmd.split())
    rep_reads = fetchRepReads(tmp_genindels_file)
    isize, smallest_indel = min([(tokFullIndel(x)[1], x) for x in rep_reads]) if len(rep_reads) > 0 else (0, '-')
    if isize > 0:
        left_trim = target_seq.find(rep_reads[smallest_indel][:10])

    os.remove(tmp_genindels_file)

    # NOTE: THIS STEP IS NOT NECESSARY, SINCE FEATURE CALCULATIONS HAVE BEEN PROVIDED ALREADY IN OLIGO DATA
    '''
    # compute features for all generated indels 
    tmp_features_file = 'tmp_features_%s_%d.txt' % (target_seq, random.randint(0, 100000))
    calculateFeaturesForGenIndelFile(tmp_genindels_file, target_seq, pam_idx - 3, tmp_features_file)
    os.remove(tmp_genindels_file)
    tmp_features, tmp_cols = readFeaturesData(tmp_features_file)
    os.remove(tmp_features_file)
    '''

    feature_data = model_df
    feature_names = list(feature_data.index)
    indels = ['_'.join(x.split('_')[2:]) for x in feature_names]
    feature_data['Indel'] = indels

    feature_columns = [x for x in feature_data.columns if
                       x not in ['Oligo ID', 'Indel', 'Left', 'Right', 'Inserted Seq']]

    for feature in feature_columns:
        if 'expThetaX' in feature_columns:
            feature_data = feature_data.drop('expThetaX', axis=1)
            feature_columns.remove('expThetaX')
        if feature not in theta_feature_columns:
            raise Exception(
                'Feature %s not found in theta file! Remove %s from feature set before proceeding' % feature)

    if len(set(theta_feature_columns).union(set(feature_columns))) != len(theta_feature_columns):
        feature_data = feature_data[['Indel'] + theta_feature_columns]
        feature_columns = theta_feature_columns

    theta_dict = {k: v for k, v in zip(theta_feature_columns, theta)}

    theta_feature_dict = {}
    for feature in feature_columns:
        if feature in theta_dict:
            theta_feature_dict[feature] = theta_dict[feature]

    theta = list(theta_feature_dict.values())

    # Predict the profile
    p_predict, _ = computePredictedProfile(feature_data, theta, feature_columns)
    in_frame, out_frame, _ = fetchIndelSizeCounts(p_predict)
    in_frame_perc = in_frame * 100.0 / (in_frame + out_frame)
    if add_null:
        p_predict['-'] = 1000
        rep_reads['-'] = target_seq[left_trim:]
    return p_predict, rep_reads, in_frame_perc


def reshapeModelOutput(repair_outcome, current_oligo):
    df_idx = []
    df_columns = []
    for sample in repair_outcome.keys():
        if 'Oligo_' + str(current_oligo) in sample:
            sample_list = sample.split('_')
            sample_idx = sample_list[0] + '_' + sample_list[1]
            if sample_idx not in df_idx:
                df_idx.append(sample_idx)

            indel = '_'.join(sample_list[2:])

            if indel not in df_columns:
                df_columns.append(indel)

    df = pd.DataFrame(index=df_idx, columns=df_columns)

    for sample, freq in repair_outcome.items():
        sample_idx = sample.split('_')[0] + '_' + sample.split('_')[1]
        sample_indel = '_'.join(sample.split('_')[2:])
        df.loc[sample_idx, sample_indel] = freq

    return df


def model(x):
    return predictionModel(x, DEFAULT_MODEL, target_seq, pam_idx)


def predictionModel(input_data, pre_trained_model, target, pam, plot=True):
    profile, rep_reads, in_frame = predictMutations(model_df, pre_trained_model, target, pam)

    repair_outcome_freqs = getAvgPreds([profile], current_oligo)
    repair_outcome_freqs_dict = {x[1]: x[0] for x in repair_outcome_freqs}

    repair_outcome_freqs = reshapeModelOutput(repair_outcome_freqs_dict, current_oligo)

    if plot:
        plotProfiles([profile], [rep_reads], [pam_idx], [False], ['Predicted'], current_oligo,
                     title='Oligo ' + str(current_oligo) + ' In Frame: %.1f%%' % in_frame)
        plt.show()

    return repair_outcome_freqs


def getSHAPValue(model, input_data, link='logit'):
    explainer = KernelExplainer(model, input_data, link=link)
    shap = explainer.shap_values(model_df)  # input data currently is train data. For this line it should be test data

    return shap


if __name__ == '__main__':
    # Note that not all oligo's in the guideset are present in the Tijsterman data present locally.
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv("FORECasT/guideset_data.txt", sep='\t')
    num_oligos = 10
    tijsterman_oligos = os.listdir('FORECasT/train/Tijsterman_Analyser')

    dfs_container = []
    oligo_data = 0
    oligo_idx = 0

    while oligo_data != 10:
        current_oligo = guideset['ID'][oligo_idx][5:]
        oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        if oligo_name not in tijsterman_oligos:
            oligo_idx += 1
            continue

        target_seq = guideset['TargetSequence'][oligo_idx]  # second index indicates oligo under inspection. if we want to consider
        # all oligo's, make for loop.
        pam_idx = guideset['PAM Index'][oligo_idx]
        feature_data = pd.read_pickle(
                "FORECasT/train/Tijsterman_Analyser/" + str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo))
        target_seq_list = [target_seq for i in range(feature_data.shape[0])]
        pam_idx_list = [pam_idx for i in range(feature_data.shape[0])]
        feature_data['TargetSequence'] = target_seq_list
        feature_data['PAMIndex'] = pam_idx_list

        model_df_temp = getKernelExplainerModelInput(feature_data, current_oligo)
        # TODO: Make sure the KernelExplainer does not remove the PAMIndex and TargetSequence columns.
        dfs_container.append(model_df_temp)
        oligo_data += 1
        oligo_idx += 1

    model_df = pd.concat(dfs_container)
    print(model_df)
    model_input = model_df.to_numpy()   # SHAP expects ndarray

    shap_values = getSHAPValue(model, model_input)
    # save shap_values to .pkl file
    with open('FORECasT/train/Tijsterman_Analyser/shap_values/' + str(guideset['ID'][19][0:5]) + '_' + str(
            current_oligo) + '.pkl', 'wb') as f:
        pkl.dump(shap_values, f)
        f.close()

    # TODO 2: Error -> MemoryError: Out of memory. This occurs because the SHAP method considers each possible coalition,
    # TODO 2: i.e. all possible combinations of features, for each sample, i.e. oligo-indel in our case. Practically this
    # TODO 2: means that we get a matrix of 393 x 3417 which takes up 25 GiB of memory

    # TODO 3: Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples to
    # TODO 3: improve computational time.
