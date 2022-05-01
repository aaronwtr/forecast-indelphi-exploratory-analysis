import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm
import config
import os
import pickle as pkl
import Lindel
import matplotlib.pyplot as plt


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


def fetch_candidate_samples():
    cand_sample_path = f"C:/Users/Aaron/Desktop/Nanobiology/MSc/MEP/interpreting-ml-based-drops/FORECasT/candidate_samples"
    dirs = os.listdir(cand_sample_path)
    candidate_samples = []

    for folder in dirs:
        files = os.listdir(f"{cand_sample_path}/{folder}")
        for file in files:
            candidate_samples.append(file)

    candidate_samples = [int(x.split('_')[1]) for x in candidate_samples]

    return candidate_samples


def get_train_data(guidedata, prereq):
    if os.path.exists(f'{config.path}/training_data.pkl'):
        train_data = pd.read_pickle(f'{config.path}/training_data.pkl')
        return train_data
    else:
        candidate_samples = fetch_candidate_samples()

        tijsterman_oligos = config.tijsterman_oligos
        ground_truth_dict = {}

        fetched_data = 0
        oligo_idx = 0
        sample_names = []
        feature_vectors = []

        label, rev_index, mh_features, frame_shift = prereq

        print('Collecting training data...')

        pbar = tqdm(total=config.dataset_size)
        while fetched_data < config.dataset_size:
            cont = False
            current_oligo = guidedata['ID'][oligo_idx][5:]
            seq = guidedata['TargetSequence'][oligo_idx]
            if int(current_oligo) not in candidate_samples and f"Oligo_{current_oligo}" in tijsterman_oligos:
                pam_idx = guidedata['PAM Index'][oligo_idx]
                nt_to_delete = int(pam_idx) - 33
                seq = seq[nt_to_delete:]
                if check_pam(seq):
                    cont = True

            oligo_idx += 1

            if cont:
                sample_names.append(f'Oligo_{current_oligo}')
                features_tmp, feature_labels = get_features(seq, mh_features)

                exp_data = pd.read_pickle(f"{config.forecast_path}/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' +
                                          str(current_oligo))

                ground_truth = exp_data['Frac Sample Reads']
                ground_truth_labels = exp_data['Indel']

                for i, indel in enumerate(ground_truth_labels):
                    if indel not in list(ground_truth_dict.keys()):
                        ground_truth_dict[indel] = [ground_truth[i]]
                    else:
                        ground_truth_dict[indel].append(ground_truth[i])

                if not len(feature_vectors):
                    feature_vectors = features_tmp
                else:
                    feature_vectors = np.append(feature_vectors, features_tmp, axis=0)
                pbar.update(1)
                fetched_data += 1

        pbar.close()

        feature_matrix = pd.DataFrame(feature_vectors, columns=feature_labels)
        feature_matrix.index = sample_names

        max_len = max([len(x) for x in ground_truth_dict.values()])
        for key in ground_truth_dict.keys():
            ground_truth_dict[key] = np.pad(ground_truth_dict[key], (0, max_len - len(ground_truth_dict[key])),
                                            'constant')

        ground_truths = pd.DataFrame(ground_truth_dict)

        return feature_matrix, ground_truths


def get_test_data(guidedata, prereq):
    if os.path.exists(f'{config.path}/test_data.pkl'):
        test_data = pd.read_pickle(f'{config.path}/training_data.pkl')
        return test_data
    else:
        candidate_samples = fetch_candidate_samples()

        tijsterman_oligos = config.tijsterman_oligos
        ground_truth_dict = {}

        fetched_data = 0
        oligo_idx = 0
        sample_names = []
        feature_vectors = []

        label, rev_index, mh_features, frame_shift = prereq

        print('Collecting testing data...')

        pbar = tqdm(total=len(candidate_samples))
        while fetched_data < len(candidate_samples):
            cont = False
            current_oligo = guidedata['ID'][oligo_idx][5:]
            seq = guidedata['TargetSequence'][oligo_idx]
            if int(current_oligo) in candidate_samples and f"Oligo_{current_oligo}" in tijsterman_oligos:
                pam_idx = guidedata['PAM Index'][oligo_idx]
                nt_to_delete = int(pam_idx) - 33
                seq = seq[nt_to_delete:]
                if check_pam(seq):
                    cont = True

            oligo_idx += 1

            if cont:
                sample_names.append(f'Oligo_{current_oligo}')
                features_tmp, feature_labels = get_features(seq, mh_features)

                exp_data = pd.read_pickle(f"{config.forecast_path}/" + str(guidedata['ID'][oligo_idx][0:5]) + '_' +
                                          str(current_oligo))

                ground_truth = exp_data['Frac Sample Reads']
                ground_truth_labels = exp_data['Indel']

                for i, indel in enumerate(ground_truth_labels):
                    if indel not in list(ground_truth_dict.keys()):
                        ground_truth_dict[indel] = [ground_truth[i]]
                    else:
                        ground_truth_dict[indel].append(ground_truth[i])

                if not len(feature_vectors):
                    feature_vectors = features_tmp
                else:
                    feature_vectors = np.append(feature_vectors, features_tmp, axis=0)
                pbar.update(1)
                fetched_data += 1

        pbar.close()

        feature_matrix = pd.DataFrame(feature_vectors, columns=feature_labels)
        feature_matrix.index = sample_names

        max_len = max([len(x) for x in ground_truth_dict.values()])
        for key in ground_truth_dict.keys():
            ground_truth_dict[key] = np.pad(ground_truth_dict[key], (0, max_len - len(ground_truth_dict[key])), 'constant')

        ground_truths = pd.DataFrame(ground_truth_dict)

        return feature_matrix, ground_truths


if __name__ == '__main__':
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))

    training_data = get_train_data(guideset, prerequesites)
    pd.to_pickle(training_data, f'{config.path}/training_data.pkl')

    test_data = get_test_data(guideset, prerequesites)
    pd.to_pickle(test_data, f'{config.path}/test_data.pkl')
