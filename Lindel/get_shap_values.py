import pandas as pd
import numpy as np
import os
import config
import pickle as pkl
import Lindel
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

'''
In this script, the Lindel pre-trained model is implemented and SHAP analysis is consequently performed on top of this
implementation.
'''


def gen_indel(sequence, cut_site):
    """This is the function that used to generate all possible unique indels and
    list the redundant classes which will be combined after"""
    nt = ['A', 'T', 'C', 'G']
    up = sequence[0:cut_site]
    down = sequence[cut_site:]
    dmax = min(len(up), len(down))
    uniqe_seq = {}
    for dstart in range(1, cut_site + 3):
        for dlen in range(1, dmax):
            if len(sequence) > dlen + dstart > cut_site - 2:
                seq = sequence[0:dstart] + sequence[dstart + dlen:]
                indel = sequence[0:dstart] + '-' * dlen + sequence[dstart + dlen:]
                array = [indel, sequence, 13, 'del', dstart - 30, dlen, None, None, None]
                try:
                    uniqe_seq[seq]
                    if dstart - 30 < 1:
                        uniqe_seq[seq] = array
                except KeyError:
                    uniqe_seq[seq] = array
    for base in nt:
        seq = sequence[0:cut_site] + base + sequence[cut_site:]
        indel = sequence[0:cut_site] + '-' + sequence[cut_site:]
        array = [sequence, indel, 13, 'ins', 0, 1, base, None, None]
        try:
            uniqe_seq[seq] = array
        except KeyError:
            uniqe_seq[seq] = array
        for base2 in nt:
            seq = sequence[0:cut_site] + base + base2 + sequence[cut_site:]
            indel = sequence[0:cut_site] + '--' + sequence[cut_site:]
            array = [sequence, indel, 13, 'ins', 0, 2, base + base2, None, None]
            try:
                uniqe_seq[seq] = array
            except KeyError:
                uniqe_seq[seq] = array
    uniq_align = label_mh(list(uniqe_seq.values()), 4)
    for read in uniq_align:
        if read[-2] == 'mh':
            merged = []
            for i in range(0, read[-1] + 1):
                merged.append((read[4] - i, read[5]))
            read[-3] = merged
    return uniq_align


def label_mh(sample, mh_len):
    '''Function to label microhomology in deletion events'''
    for k in range(len(sample)):
        read = sample[k]
        if read[3] == 'del':
            idx = read[2] + read[4] + 17
            idx2 = idx + read[5]
            x = mh_len if read[5] > mh_len else read[5]
            for i in range(x, 0, -1):
                if read[1][idx - i:idx] == read[1][idx2 - i:idx2] and i <= read[5]:
                    sample[k][-2] = 'mh'
                    sample[k][-1] = i
                    break
            if sample[k][-2] != 'mh':
                sample[k][-1] = 0
    return sample


def create_mh_feature_array(ft, uniq_indels):
    '''Used to create microhomology feature array
       require the features and label
    '''
    ft_array = np.zeros(len(ft))
    for read in uniq_indels:
        if read[-2] == 'mh':
            mh = str(read[4]) + '+' + str(read[5]) + '+' + str(read[-1])
            try:
                ft_array[ft[mh]] = 1
            except KeyError:
                pass
        else:
            pt = str(read[4]) + '+' + str(read[5]) + '+' + str(0)
            try:
                ft_array[ft[pt]] = 1
            except KeyError:
                pass
    return ft_array


def onehotencoder(seq):
    '''convert to single and di-nucleotide hotencode'''
    nt = ['A', 'T', 'C', 'G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i] + str(k))
    for k in range(l - 1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i] + nt[j] + str(k))
    head_idx = {}
    for idx, key in enumerate(head):
        head_idx[key] = idx
    head_idx_keys = list(head_idx.keys())
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j] + str(j)]] = 1.
    for k in range(l - 1):
        encode[head_idx[seq[k:k + 2] + str(k)]] = 1.
    encode_dict = dict(zip(head_idx_keys, encode))
    return encode_dict


def check_pam(seq):
    pam = ['AGG', 'TGG', 'CGG', 'GGG']

    if seq[33:36] in pam:
        return True
    else:
        return False


def gen_prediction(seq, wb, prereq):
    '''generate the prediction for all classes, redundant classes will be combined'''
    pam = {'AGG': 0, 'TGG': 0, 'CGG': 0, 'GGG': 0}
    guide = seq[13:33]
    if seq[33:36] not in pam:
        return 'Error: No PAM sequence is identified.'

    w1, b1, w2, b2, w3, b3 = wb
    print(len(w1))
    label, rev_index, features, frame_shift = prereq
    indels = gen_indel(seq, 30)
    input_indel = np.array(list(onehotencoder(guide).values()))
    input_ins = np.array(list(onehotencoder(guide[-6:]).values()))
    input_del = np.concatenate((create_mh_feature_array(features, indels), input_indel), axis=None)
    print(type(input_del))
    print(len(input_del))
    print(len(input_ins))

    cmax = gen_cmatrix(indels, label)  # combine redundant classes
    dratio, insratio = softmax(np.dot(input_indel, w1) + b1)
    ds = softmax(np.dot(input_del, w2) + b2)
    ins = softmax(np.dot(input_ins, w3) + b3)
    y_hat = np.concatenate((ds * dratio, ins * insratio), axis=None) * cmax

    return y_hat, np.dot(y_hat, frame_shift)


def softmax(weights):
    return np.exp(weights) / sum(np.exp(weights))


def gen_cmatrix(indels, label):
    ''' Combine redundant classes based on microhomology, matrix operation'''
    combine = []
    for s in indels:
        if s[-2] == 'mh':
            tmp = []
            for k in s[-3]:
                try:
                    tmp.append(label['+'.join(list(map(str, k)))])
                except KeyError:
                    pass
            if len(tmp) > 1:
                combine.append(tmp)
    temp = np.diag(np.ones(557), 0)
    for key in combine:
        for i in key[1:]:
            temp[i, key[0]] = 1
            temp[i, i] = 0
    return sparse.csr_matrix(temp)


def get_features(seq, features):
    guide = seq[13:33]

    features_dict = {f'{k}_MH': v for k, v in features.items()}

    indels = gen_indel(seq, 30)

    input_del_dict = onehotencoder(guide)
    input_del = list(input_del_dict.values())

    input_ins_dict = onehotencoder(guide[-6:])
    input_ins_dict = {k + '_ins': v for k, v in input_ins_dict.items()}
    input_ins = np.array(list(input_ins_dict.values()))
    input_ins = input_ins.reshape(1, len(input_ins))

    mh_ft_array = create_mh_feature_array(features, indels)

    input_del = np.concatenate((mh_ft_array, input_del), axis=None)
    features_keys = list(features_dict.keys())
    features_keys.extend(list(input_del_dict.keys()))
    features_keys.extend(list(input_ins_dict.keys()))
    feature_labels = np.array(features_keys)
    input_del = input_del.reshape(1, len(input_del))

    input_del_arr = np.array(input_del[0])
    input_ins_arr = np.array(input_ins[0])
    features = np.append(input_del_arr, input_ins_arr).reshape(1, -1)
    features = pd.DataFrame(features, columns=features_keys)

    features = features.to_numpy()

    return features, feature_labels


def getExplanationData(guidedata, ioi, prereq):
    oligo_of_interest = int(ioi.split('_')[1])

    for index, row in guideset.iterrows():
        if oligo_of_interest == int(row['ID'][5:]):
            oligo_idx = index + 1
            pam_idx = row['PAM Index']
            nt_to_delete = pam_idx - 33  # We need to make sure the PAM is at the 33 idx
            seq = row['TargetSequence'][nt_to_delete:]
            break

    label, rev_index, mh_features, frame_shift = prereq

    features, feature_labels = get_features(seq, mh_features)
    sample_names = [f'Oligo_{oligo_of_interest}']

    # Note that we have now stored ioi deletion in first row and ioi insertion in the second row

    oligo_data = features.shape[0]

    print('Collecting explanation data...')
    explanation_data = features.copy()
    pbar = tqdm(total=config.dataset_size)
    while oligo_data < config.dataset_size:
        current_oligo = guidedata['ID'][oligo_idx][5:]
        cont = True

        for index, row in guideset.iterrows():
            if current_oligo == row['ID'][5:]: # should not be oligo of interest!
                oligo_idx = index + 1
                pam_idx = row['PAM Index']
                nt_to_delete = pam_idx - 33  # We need to make sure the PAM is at the 33 idx
                seq = row['TargetSequence'][nt_to_delete:]
                if check_pam(seq):
                    break
                else:
                    cont = False

        if cont:
            sample_names.append(f'Oligo_{current_oligo}')
            features_tmp, feature_labels = get_features(seq, mh_features)
            explanation_data = np.concatenate((explanation_data, features_tmp))
            pbar.update(1)
            oligo_data = explanation_data.shape[0]

    pbar.close()

    explanation_data = pd.DataFrame(explanation_data, columns=feature_labels)
    explanation_data.index = sample_names

    return explanation_data


if __name__ == '__main__':
    weights = pkl.load(open(os.path.join(Lindel.__path__[0], "Model_weights.pkl"), 'rb'))
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')

    explanation_dataset_path = f'{config.path}/explanation_datasets/dataset_size_{config.dataset_size}'
    explanation_dataset_name = f'{config.repair_outcome_of_interest}.pkl'

    if os.path.isfile(f'{explanation_dataset_path}/{explanation_dataset_name}'):
        explanation_df = pd.read_pickle(f'{explanation_dataset_path}/{explanation_dataset_name}')
    else:
        explanation_df = getExplanationData(guideset, config.repair_outcome_of_interest, prerequesites)
        explanation_df.to_pickle(f'{explanation_dataset_path}/{explanation_dataset_name}')

    print('Done!')
    seq = 'GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCAT'
    # filename = config.sample_of_interest.split('_')[0:2]
    # filename = '_'.join(filename)
    y_hat, fs = gen_prediction(seq, weights, prerequesites)
    # filename += '_fs=' + str(round(fs, 3)) + '.txt'
    # rev_index = prerequesites[1]
    # pred_freq = {}
    # for i in range(len(y_hat)):
    #     if y_hat[i] != 0:
    #         pred_freq[rev_index[i]] = y_hat[i]
    # pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1], reverse=True)

    # TODO 1 Wrap Lindel prediction model in a function that that takes in the explanation dataset and is able to compute
    # TODO 1 the output for the model.
        # TODO 1A Rewrite the prediction function to take explanation dataset and perform the prediction based on a single
        # TODO 1A feature vector rather than separate feature vectors for deletions and insertions.
