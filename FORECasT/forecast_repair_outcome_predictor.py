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
import config
from tqdm import tqdm

from predictor.features import calculateFeaturesForGenIndelFile, readFeaturesData
from predictor.model import readTheta, computePredictedProfile
from predictor.predict import predictMutations, INDELGENTARGET_EXE, fetchRepReads
from selftarget.indel import tokFullIndel
from selftarget.profile import fetchIndelSizeCounts
from selftarget.view import plotProfiles, getAvgPreds

'''
In this script, the FORECasT pre-trained model is implemented and can be used to obtain repair profiles given an oligo
dataset, target sequence and PAM location index.
'''


def predictMutations(input_data, theta_file, target, pam, add_null=True):
    feature_matrix = input_data
    feature_columns = input_data.columns
    theta, train_set, theta_feature_columns = readTheta(theta_file)

    theta_dict = {k: v for k, v in zip(theta_feature_columns, theta)}
    theta_feature_dict = {}
    for feature in feature_columns:
        if feature in theta_dict:
            theta_feature_dict[feature] = theta_dict[feature]

    theta_feature_columns = [feature for feature in theta_feature_columns if feature in feature_columns]
    theta = list(theta_feature_dict.values())

    left_trim = 0
    tmp_genindels_file = 'tmp_genindels_%s_%d.txt' % (target, random.randint(0, 100000))
    cmd = INDELGENTARGET_EXE + ' %s %d %s' % (target, pam, tmp_genindels_file)

    subprocess.check_call(cmd.split())
    rep_reads = fetchRepReads(tmp_genindels_file)
    isize, smallest_indel = min([(tokFullIndel(x)[1], x) for x in rep_reads]) if len(rep_reads) > 0 else (0, '-')

    if isize > 0:
        left_trim = target_seq.find(rep_reads[smallest_indel][:10])

    os.remove(tmp_genindels_file)

    if len(set(theta_feature_columns).difference(set(feature_columns))) != 0:
        raise Exception('Stored feature names associated with model thetas are not contained in those computed')

    if len(set(theta_feature_columns).union(set(feature_columns))) != len(theta_feature_columns):
        feature_matrix = feature_matrix[['Indel'] + theta_feature_columns]

    # feature_matrix.set_index('Indel', inplace=True)
    # Predict the profile
    p_predict, _ = computePredictedProfile(feature_matrix, theta, theta_feature_columns)
    in_frame, out_frame, _ = fetchIndelSizeCounts(p_predict)
    in_frame_perc = in_frame * 100.0 / (in_frame + out_frame)
    if add_null:
        p_predict['-'] = 1000
        rep_reads['-'] = target_seq[left_trim:]

    return p_predict, rep_reads, in_frame_perc


def model(x):
    return predictionModel(x, DEFAULT_MODEL, target_seq, pam_idx, num_plots)


def predictionModel(input_data, pre_trained_model, target, pam, num_plot, plot=False):
    profile, rep_reads, in_frame = predictMutations(input_data, pre_trained_model, target, pam)

    profile_freqs = list(profile.values())
    profile_freqs.sort(reverse=True)
    profile_freqs = profile_freqs[1:]

    # if profile_freqs[0] - profile_freqs[1] < 0.1:
    #     plot = True

    plot = True

    if plot:
        plotProfiles([profile], [rep_reads], [pam_idx], [False], ['Predicted'], current_oligo,
                     title='Oligo ' + str(current_oligo) + ' In Frame: %.1f%%' % in_frame)
        plt.savefig("FORECasT/repair_outcomes/oligo_%s.pdf" % current_oligo)
        plt.close()
        plot = False

        return 1

    return 0


if __name__ == '__main__':
    # Note that not all oligo's in the guideset are present in the Tijsterman data present locally.
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv("FORECasT/guideset_data.txt", sep='\t')
    tijsterman_oligos = os.listdir('FORECasT/train/Tijsterman_Analyser')
    DEFAULT_MODEL = config.DEFAULT_MODEL

    dfs_container = []
    oligo_idx = 0

    target_seq_list = []
    pam_idx_list = []
    oligos_list = []
    indels = []
    pred_data = []
    data_found = False
    num_plots = 0

    pbar = tqdm(total=100)
    while num_plots != 100:
        while not data_found:
            current_oligo = guideset['ID'][oligo_idx][5:]
            oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
            if oligo_name not in tijsterman_oligos:
                oligo_idx += 1
                continue
            data_found = True

        current_oligo = guideset['ID'][oligo_idx][5:]
        oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)

        target_seq = guideset['TargetSequence'][oligo_idx]
        pam_idx = guideset['PAM Index'][oligo_idx]
        feature_data = pd.read_pickle("FORECasT/train/Tijsterman_Analyser/" + oligo_name)

        num_plots += model(feature_data)
        pbar.update(1)
        oligo_idx += 1
        data_found = False

    pbar.close()
