import pandas as pd
import pickle as pkl
import config
import plotly.express as px
from warnings import simplefilter
import os
from tqdm import tqdm
import numpy as np
from Lindel_prediction import predict_single_sample


def KL(p1, p2, ignore_null=True, missing_count=0.5):
    p1_indels = set([x for x in p1 if p1[x] > 0 and (x != '-' or not ignore_null)])
    p2_indels = set([x for x in p2 if p2[x] > 0 and (x != '-' or not ignore_null)])
    common = p1_indels.intersection(p2_indels)
    p1_only = p1_indels.difference(p2_indels)
    p2_only = p2_indels.difference(p1_indels)

    p1_total = sum([p1[x] for x in p1_indels]) + missing_count * len(p2_only)
    p2_total = sum([p2[x] for x in p2_indels]) + missing_count * len(p1_only)

    if p1_total > 0 and p2_total > 0:
        norm1, norm2 = 1.0 / p1_total, 1.0 / p2_total
        score = 0.0
        for indel in common:
            score += p1[indel] * norm1 * np.log2(p1[indel] * norm1 / (p2[indel] * norm2))
        for indel in p1_only:
            score += p1[indel] * norm1 * np.log2(p1[indel] * norm1 / (missing_count * norm2))
        for indel in p2_only:
            score += missing_count * norm1 * np.log2(missing_count * norm1 / (p2[indel] * norm2))
    else:
        score = np.nan
    return score


def symmetricKL(profile1, profile2, ignore_null=True):
    return 0.5 * KL(profile1, profile2, ignore_null) + 0.5 * KL(profile2, profile1, ignore_null)


def generate_boxplot():
    with open(f'{config.path}/kl_divs/kl_divs_N={config.performance_samples}.pkl', 'rb') as f:
        kl_divs_forecast = pkl.load(f)
    f.close()

    with open(f'{config.path}/kl_divs/kl_divs_baseline_N=1e03.pkl', 'rb') as f:
        kl_divs_baseline = pkl.load(f)
    f.close()

    with open(f'C:/Users/Aaron/Desktop/Nanobiology/MSc/MEP/interpreting-ml-based-drops/Lindel/kl_divs/kl_divs_N=1e03.pkl', 'rb') as f:
        kl_divs_lindel = pkl.load(f)
    f.close()

    with open(f'{config.path}/kl_divs_N=1e03_trained.pkl', 'rb') as f:
        kl_divs_lindel_trained = pkl.load(f)
    f.close()

    forecast_values = [kl_divs_forecast[x] for x in kl_divs_forecast]
    forecast_dict = {'FORECasT': forecast_values}

    baseline_values = [kl_divs_baseline[x] for x in kl_divs_baseline]
    baseline_dict = {'Baseline': baseline_values}

    lindel_values = [kl_divs_lindel[x] for x in kl_divs_lindel]
    lindel_dict = {'Lindel': lindel_values}

    lindel_trained_values = [kl_divs_lindel_trained[x] for x in kl_divs_lindel_trained]
    lindel_trained_dict = {'Lindel_trained': lindel_trained_values}

    df = pd.DataFrame(baseline_dict)
    df = df.append(pd.DataFrame(forecast_dict))
    df = df.append(pd.DataFrame(lindel_dict))
    df = df.append(pd.DataFrame(lindel_trained_dict))
    df = df.melt(var_name='Model')

    df.rename(columns={'value': 'KL divergence'}, inplace=True)
    fig = px.box(df, x="Model", y="KL divergence", color="Model",
                 points='all')

    fig.update_layout(title_text="Performance as measured by KL divergence on FORECasT data (N=1000)")

    fig.show()


if __name__ == '__main__':
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    tijsterman_oligos = os.listdir(f'{config.forecast_path}')
    test_data = pkl.load(open(f'{config.path}/test_data.pkl', 'rb'))
    training_data = pkl.load(open(f'{config.path}/training_data.pkl', 'rb'))


    dfs_container = []
    oligo_idx = 0

    target_seq_list = []
    pam_idx_list = []
    oligos_list = []
    indels = []
    pred_data = []
    data_found = False
    num_samples = 0

    kl_divs = {}
    analyze = False

    test_tijsterman_oligos = config.tmp_test_tijsterman_oligos

    fetched_data = 0
    oligo_idx = 0
    sample_names = []
    feature_vectors = []

    data_getter = len(test_tijsterman_oligos)
    data_count = 0

    if not analyze:
        pbar = tqdm(total=len(test_tijsterman_oligos))
        while data_count < len(test_tijsterman_oligos):
            cont = False
            data_count += 1
            current_oligo = guideset['ID'][oligo_idx][5:]
            seq = guideset['TargetSequence'][oligo_idx]
            while not data_found:
                cont = True
                current_oligo = guideset['ID'][oligo_idx][5:]
                oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)
                if oligo_name not in test_tijsterman_oligos:
                    oligo_idx += 1
                    continue
                data_found = True

            current_oligo = guideset['ID'][oligo_idx][5:]
            oligo_name = str(guideset['ID'][oligo_idx][0:5]) + '_' + str(current_oligo)

            target_seq = guideset['TargetSequence'][oligo_idx]
            pam_idx = guideset['PAM Index'][oligo_idx]
            feature_data = pd.read_pickle(f"{config.tmp_test_forecast_path}/" + oligo_name)
            experimental_distribution = feature_data['Frac Sample Reads']

            experimental_distribution = dict(zip(feature_data['Indel'], experimental_distribution))
            experimental_distribution = dict(sorted(experimental_distribution.items(), key=lambda x: x[0]))

            try:
                predicted_distribution = predict_single_sample(current_oligo, guideset, data=test_data)
            except KeyError:
                data_found = False
                cont = False
                oligo_idx += 1

            if cont:
                if predicted_distribution != 0:
                    predicted_distribution = dict(sorted(predicted_distribution.items(), key=lambda x: x[0]))
                    oligo_idx += 1
                    data_found = False
                    num_samples += 1

                    KL_div = symmetricKL(experimental_distribution, predicted_distribution)
                    kl_divs[oligo_name] = KL_div
                    pbar.update(1)

                else:
                    oligo_idx += 1
                    data_found = False

        pbar.close()

        with open(f'{config.path}/kl_divs_N={data_count}_trained.pkl', 'wb') as f:
            pkl.dump(kl_divs, f)
    else:
        with open(f'{config.path}/kl_divs/kl_divs_all_test_samples_lindel.pkl', 'rb') as f:
            kl_divs = pkl.load(f)
            kl_divs_list = list(kl_divs.values())
            mean_kl_div = np.mean(kl_divs_list)
            print(mean_kl_div)
            generate_boxplot()
