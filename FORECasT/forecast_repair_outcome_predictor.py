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
        left_trim = target.find(rep_reads[smallest_indel][:10])

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
        rep_reads['-'] = target[left_trim:]

    return p_predict, rep_reads, in_frame_perc


class RepairOutcomeGenerator:
    def __init__(self):
        self.current_oligo = None
        self.pam_idx = None
        self.target_seq = None
        self.folder = None

    def model(self, x):
        return ROG.predictionModel(x, DEFAULT_MODEL, self.target_seq, self.pam_idx)

    def predictionModel(self, input_data, pre_trained_model, target, pam, plot=False):
        profile, rep_reads, in_frame = predictMutations(input_data, pre_trained_model, target, pam)

        profile_freqs = list(profile.values())
        profile_freqs.sort(reverse=True)
        profile_freqs = profile_freqs[1:]

        # if profile_freqs[0] - profile_freqs[1] < 0.1:
        #     plot = True

        plot = True

        if plot:
            pp = plotProfiles([profile], [rep_reads], [self.pam_idx], [False], ['Predicted'], self.current_oligo,
                              title='Oligo ' + str(self.current_oligo) + ' In Frame: %.1f%%' % in_frame)
            if pp == 0:
                return 0

            plt.savefig(f"FORECasT/repair_outcomes/{self.folder}/oligo_{self.current_oligo}.pdf")
            plt.close()
            plot = False

            return 1

        return 0

    def get_repair_outcome_profiles(self):
        oligo_idx = 0

        data_found = False
        num_plots = 0

        pbar = tqdm(total=10)
        while num_plots != 10:
            while not data_found:
                current_oligo = guideset['ID'][oligo_idx][5:]
                oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
                if oligo_name not in tijsterman_oligos:
                    oligo_idx += 1
                    continue
                data_found = True

            self.current_oligo = guideset['ID'][oligo_idx][5:]
            oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)

            self.target_seq = guideset['TargetSequence'][oligo_idx]
            self.pam_idx = guideset['PAM Index'][oligo_idx]
            feature_data = pd.read_pickle("FORECasT/train/Tijsterman_Analyser/" + oligo_name)

            num_plots += self.model(feature_data)
            if self.model(feature_data) is not 0:
                pbar.update(1)
            oligo_idx += 1
            data_found = False

        pbar.close()

    def get_candidate_repair_profiles(self):
        sample_types = os.listdir(config.candidate_samples)

        for sample_type in sample_types:
            self.folder = sample_type
            folder_contents = os.listdir(f"{config.candidate_samples}/{sample_type}")
            if len(folder_contents) == 1:
                continue
            oligo_idx = 0
            cont = False
            data_found = False
            num_plots = 0
            print(f'{config.candidate_samples}/{sample_type}')
            candidate_samples = os.listdir(f'{config.candidate_samples}/{self.folder}')
            while not data_found:
                current_oligo = guideset['ID'][oligo_idx][5:]
                oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
                # print(candidate_samples)
                if oligo_name in candidate_samples:
                    cont = True
                else:
                    oligo_idx += 1

                if cont:
                    self.current_oligo = guideset['ID'][oligo_idx][5:]
                    oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)

                    self.target_seq = guideset['TargetSequence'][oligo_idx]
                    self.pam_idx = guideset['PAM Index'][oligo_idx]
                    feature_data = pd.read_pickle(f"{config.hd_test_path}/{oligo_name}")

                    self.model(feature_data)
                    num_plots += 1
                    oligo_idx += 1
                    cont = False

                if num_plots == len(candidate_samples):
                    data_found = True


if __name__ == '__main__':
    # Note that not all oligo's in the guideset are present in the Tijsterman data present locally.
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv("FORECasT/guideset_data.txt", sep='\t')
    # tijsterman_oligos = os.listdir('FORECasT/train/Tijsterman_Analyser')
    tijsterman_oligos = config.hd_test_tijsterman_oligos
    DEFAULT_MODEL = config.DEFAULT_MODEL

    ROG = RepairOutcomeGenerator()
    ROG.get_candidate_repair_profiles()
