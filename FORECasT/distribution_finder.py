import pandas as pd
import config
from tqdm import tqdm
import os
import numpy as np
import pickle as pkl
import shutil


def generating_average_distribution():
    oligo_idx = 0

    data_found = False
    num_samples = 0

    experimental_distribution_dict = {}

    pbar = tqdm(total=int(float(config.performance_samples)))
    while num_samples != int(float(config.performance_samples)):
        while not data_found:
            current_oligo = guideset['ID'][oligo_idx][5:]
            oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
            if oligo_name not in tijsterman_oligos:
                oligo_idx += 1
                continue
            data_found = True

        current_oligo = guideset['ID'][oligo_idx][5:]
        oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        oligo_path = f"{config.path}/train/Tijsterman_Analyser/" + oligo_name

        feature_data = pd.read_pickle(oligo_path)
        experimental_distribution = feature_data['Frac Sample Reads']

        experimental_distribution = dict(zip(feature_data['Indel'], experimental_distribution))

        for indel in feature_data['Indel']:
            if indel not in experimental_distribution_dict:
                experimental_distribution_dict[indel] = []
            experimental_distribution_dict[indel].append(experimental_distribution[indel])

        print(experimental_distribution)
        pbar.update(1)
        oligo_idx += 1
        num_samples += 1
        data_found = False

    experimental_distribution_df = pd.DataFrame.from_dict(experimental_distribution_dict, orient='index').transpose()
    experimental_distribution_df = experimental_distribution_df.fillna(0)

    freqs_dict = {}
    for indel in experimental_distribution_df.columns:
        freqs_dict[indel] = experimental_distribution_df[indel].mean()

    with open(f"baseline_distribution.pkl", 'wb') as f:
        pkl.dump(freqs_dict, f)


def select_candidate_samples(filtered_samples):
    oligo_idx = 0

    data_found = False
    num_samples = 0

    experimental_distribution_dict = {}

    pbar = tqdm(total=len(tijsterman_oligos))
    count = 0
    while num_samples != len(tijsterman_oligos):
        while not data_found:
            current_oligo = guideset['ID'][oligo_idx][5:]
            oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
            if oligo_name not in tijsterman_oligos:
                oligo_idx += 1
                continue
            data_found = True

        threshold = 0.1

        current_oligo = guideset['ID'][oligo_idx][5:]
        oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
        oligo_path = "E:/Aaron/Nanobiology/MSc/Year3/MEP/test/Tijsterman_Analyser/" + oligo_name
        # candidate_path = f"{config.path}/candidate_samples/test_data/large_deletions_freq_{int(threshold * 100)}+/" + \
        #                 oligo_name

        candidate_path = f"E:/Aaron/Nanobiology/MSc/Year3/MEP/candidate_samples/test_data/single_nucleotide_insertions"\
                f"_freq_{int(threshold * 100)}+/" + oligo_name

        feature_data = pd.read_pickle(oligo_path)
        experimental_distribution = feature_data['Frac Sample Reads']

        experimental_distribution = dict(zip(feature_data['Indel'], experimental_distribution))
        experimental_distribution = dict(sorted(experimental_distribution.items(), key=lambda x: x[1], reverse=True))

        indels = list(experimental_distribution.keys())
        first_freq = experimental_distribution[list(experimental_distribution.keys())[0]]

        indel_strip = indels[0].split('_')[0]
        pbar.update(1)
        if oligo_name in filtered_samples and indel_strip[0] == 'I' and int(indel_strip[1:]) == 1 and first_freq >= \
                threshold:
            shutil.copyfile(oligo_path, candidate_path)
            num_samples += 1

        oligo_idx += 1
        count += 1
        data_found = False


if __name__ == '__main__':
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    tijsterman_oligos = config.hd_test_tijsterman_oligos
    DEFAULT_MODEL = config.DEFAULT_MODEL
    # open filtered_oligos.pkl
    with open(f"{config.path}/filtered_oligos.pkl", 'rb') as f:
        filtered_oligos = pkl.load(f)

    select_candidate_samples(filtered_oligos)

    # candidate_samples_large_del = os.listdir(f'{config.path}/candidate_samples/single_nucleotide_insertions_freq_50+/')
    # for sample in candidate_samples_large_del:
    #     sample_path = f"{config.path}/candidate_samples/single_nucleotide_insertions_freq_50+/{sample}"
    #     sample_data = pd.read_pickle(sample_path)
