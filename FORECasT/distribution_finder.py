import pandas as pd
import config
from tqdm import tqdm
import os
import numpy as np
import pickle as pkl

guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
tijsterman_oligos = os.listdir(f'{config.path}/train/Tijsterman_Analyser')
DEFAULT_MODEL = config.DEFAULT_MODEL

oligo_idx = 0

data_found = False
num_samples = 0

kl_divs = {}
analyze = False
random = True


all_freqs = []
current_oligo = guideset['ID'][oligo_idx][5:]
oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
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

    target_seq = guideset['TargetSequence'][oligo_idx]
    pam_idx = guideset['PAM Index'][oligo_idx]
    feature_data = pd.read_pickle(f"{config.path}/train/Tijsterman_Analyser/" + oligo_name)
    experimental_distribution = feature_data['Frac Sample Reads']

    experimental_distribution = dict(zip(feature_data['Indel'], experimental_distribution))

    for indel in feature_data['Indel']:
        if indel not in experimental_distribution_dict:
            experimental_distribution_dict[indel] = []
        experimental_distribution_dict[indel].append(experimental_distribution[indel])

    # freqs = list(experimental_distribution.values())
    # freqs = sorted(freqs, reverse=True)
    # all_freqs.append(freqs)
    pbar.update(1)
    oligo_idx += 1
    num_samples += 1
    data_found = False

experimental_distribution_df = pd.DataFrame.from_dict(experimental_distribution_dict, orient='index').transpose()
experimental_distribution_df = experimental_distribution_df.fillna(0)

freqs_dict = {}
for indel in experimental_distribution_df.columns:
    freqs_dict[indel] = experimental_distribution_df[indel].mean()

# save the dictionary to a pickle file
with open(f"baseline_distribution.pkl", 'wb') as f:
    pkl.dump(freqs_dict, f)
