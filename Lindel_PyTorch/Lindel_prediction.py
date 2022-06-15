import pickle as pkl
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random
import subprocess
import torch
import os

import Lindel
import config
from Lindel.Predictor import *
from get_shap_values import check_pam
from model import *
from predictor.predict import INDELGENTARGET_EXE, fetchRepReads
from selftarget.view import plotProfiles


def predict_all_samples_legacy(save=False, pretrained=False, **kwargs):
    if 'data' in kwargs:
        x, out_data = kwargs['data']
        pretrained = False
        weights = torch.load(open(f'{config.path}/model_params/model_params_344_epochs_1e-05_weight_decay.pkl', 'rb'))
    else:
        weights = pre_trained_weights

    for i in tqdm(range(guideset.shape[0])):
        cont = False
        current_sample = guideset.iloc[i]['ID']
        sample_idx = int(current_sample[5:])
        for index, row in guideset.iterrows():
            if sample_idx == index:
                pam_idx = row['PAM Index']
                nt_to_delete = pam_idx - 33  # We need to make sure the PAM is at the 33 idx
                seq = row['TargetSequence'][nt_to_delete:]
                if check_pam(seq):
                    break
                else:
                    cont = True
        if cont:
            continue

        filename = current_sample.split('_')[0:2]
        filename = '_'.join(filename)
        if pretrained:
            y_hat, fs = gen_prediction(seq, weights, prerequesites)
            filename += '_fs=' + str(round(fs, 3)) + '.txt'
        else:
            test_data, _ = pkl.load(open(f'{config.path}/test_data.pkl', 'rb'))
            test_data = test_data.values
            test_data = torch.tensor(test_data, dtype=torch.float)

            model = LogisticRegression(test_data[1], 1)
            model.load_state_dict(weights)

            y_hat = model()

        rev_index = prerequesites[1]
        pred_freq = {}
        for j in range(len(y_hat)):
            if y_hat[j] != 0:
                pred_freq[rev_index[j]] = y_hat[j]
        pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1], reverse=True)
        write_file(seq, pred_sorted, pred_freq, filename)

        path_to_file = f'repair_outcomes/cache/{filename}'
        current_repair_outcome = open_file(path_to_file)
        column_names = ['Target sequence outcome', 'Integration frequency', 'Indel label']
        current_repair_outcome.columns = column_names
        indel_name = current_repair_outcome.iloc[0]['Indel label']
        indel_name = indel_name.split('+')[0]
        indel_name = indel_name.split(' ')[0]
        indel_length = int(indel_name[1:])

        if save:
            if indel_name[0] == 'D' and indel_length >= 15:
                np.savetxt(f'repair_outcomes/candidate_repair_outcomes/deletions/{current_sample}_{indel_name}.txt',
                           current_repair_outcome.values, fmt='%s', delimiter='\t')
            else:
                np.savetxt(f'repair_outcomes/low_freq_repair_outcomes/{current_sample}_{indel_name}',
                           current_repair_outcome.values, fmt='%s', delimiter='\t')

    return


def profilePlotter(profile, rep_reads, pam, oligo_idx, plot=True):
    profile_freqs = list(profile.values())
    profile_freqs.sort(reverse=True)

    if plot:
        pp = plotProfiles([profile], [rep_reads], [pam], [False], ['Predicted'], oligo_idx,
                          title='Oligo ' + str(oligo_idx))
        if pp == 0:
            return 0

        plt.savefig(f"repair_outcomes/candidate_repair_outcomes/deletions/'Oligo_{oligo_idx}_{config.repair_outcome_of_interest}.pdf")
        plt.close()
        plot = False

        return 1

    return 0


def predict_single_sample(oligo_idx, guideset, save=True, pretrained=False, **kwargs):
    if 'data' in kwargs:
        x, out_data = kwargs['data']
        pretrained = False
        weights = torch.load(open(f'{config.path}/model_params/model_params_344_epochs_1e-05_weight_decay.pkl', 'rb'))
    else:
        weights = pre_trained_weights

    cont = False
    for index, row in guideset.iterrows():
        if int(oligo_idx) == index:
            cont = True
            oligo_name = row['ID'][5:]
            pam_idx = row['PAM Index']
            nt_to_delete = pam_idx - 33  # We need to make sure the PAM is at the 33 idx
            seq = row['TargetSequence'][nt_to_delete:]
            if check_pam(seq):
                break
            else:
                return 0

    if cont:
        filename = f'Oligo_{oligo_name}'
        if pretrained:
            y_hat, fs = gen_prediction(seq, weights, prerequesites)
            filename += '_fs=' + str(round(fs, 3)) + '.txt'

            rev_index = prerequesites[1]
            pred_freq = {}
            for i in range(len(y_hat)):
                if y_hat[i] != 0:
                    pred_freq[rev_index[i]] = y_hat[i]
            pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1], reverse=True)
            write_file(seq, pred_sorted, pred_freq, filename)
            path_to_file = f'repair_outcomes/cache/{filename}'
            current_repair_outcome = open_file(path_to_file)

            column_names = ['Target sequence outcome', 'Integration frequency', 'Indel label']
            current_repair_outcome.columns = column_names
            indel_names = list(current_repair_outcome['Indel label'])
            int_freqs = list(current_repair_outcome['Integration frequency'])

            indel_names = [x.split('+')[0] for x in indel_names]
            indel_names = [x.split('  ')[0] for x in indel_names]

            indels_done = []
            for indel in indel_names:
                count = 0
                if indel not in indels_done:
                    count += 1
                    for i in range(len(indel_names)):
                        if indel_names[i] == indel:
                            indel_names[i] = indel + '_' + str(count)
                            indel_names[i] = indel_names[i].split('_')[0] + '_' + indel_names[i].split('_')[1]
                            count += 1
                indels_done.append(indel)

            pred_freq = {}
            for i in range(len(int_freqs)):
                if int_freqs[i] != 0:
                    pred_freq[indel_names[i]] = int_freqs[i] / 100
            pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1], reverse=True)

            pred_sorted = {k: v for k, v in pred_sorted}

        else:
            cols = list(out_data.columns.values)
            in_features = x.shape[1]
            out_features = out_data.shape[1]

            print(f'Oligo_{oligo_name}')

            x = x.loc[f'Oligo_{oligo_name}']
            x = x.values
            x = torch.tensor(x, dtype=torch.float)

            model = LogisticRegression(in_features, out_features)
            model.load_state_dict(weights)

            y_hat = model(x).tolist()
            y_hat = [round(x * 10, 3) for x in y_hat]
            y_hat = softmax(y_hat)

            pred_freq = {}
            for i in range(len(y_hat)):
                pred_freq[cols[i]] = y_hat[i]

            pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1], reverse=True)
            pred_sorted = {k: v for k, v in pred_sorted}
            pred_sorted = {k: v for k, v in pred_sorted.items() if not k.startswith('Indel_')}
            print(pred_sorted)

            # tmp_genindels_file = 'tmp_genindels_%s_%d.txt' % (seq, random.randint(0, 100000))
            # cmd = INDELGENTARGET_EXE + ' %s %d %s' % (seq, pam_idx, tmp_genindels_file)
            #
            # subprocess.check_call(cmd.split())
            # rep_reads = fetchRepReads(tmp_genindels_file)
            #
            # profilePlotter(pred_sorted, rep_reads, pam_idx, oligo_name)

        return pred_sorted


if __name__ == '__main__':
    pre_trained_weights = pkl.load(open(os.path.join(Lindel.__path__[0], "Model_weights.pkl"), 'rb'))
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    test_data = pkl.load(open(f'{config.path}/test_data.pkl', 'rb'))

    oligo_tmp = test_data[0]
    oligo_row = oligo_tmp.loc['Oligo_19628']
    oligo_row = oligo_tmp.loc['Oligo_4698']

    # oligo_4698 = predict_single_sample(2664, guideset, data=test_data)
    oligo_181 = predict_single_sample(6365, guideset, data=test_data)

    # oligo_4698 = predict_single_sample(2664, guideset, pretrained=True)
    # print(oligo_4698)

    # predicted_freqs = []
    # for idx in tqdm(oligos_idx):
    #     predicted_freqs.append(predict_single_sample(idx, guideset, data=test_data))
    #     # predicted_freqs is a list of dicts for the specified samples
