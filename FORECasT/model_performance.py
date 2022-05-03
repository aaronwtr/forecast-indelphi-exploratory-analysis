import pandas as pd
import pickle as pkl
import config
from forecast_repair_outcome_predictor import predictMutations
from warnings import simplefilter
import os
from tqdm import tqdm
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter


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


def model(x):
    return predictionModel(x, DEFAULT_MODEL, target_seq, pam_idx, num_samples)


def predictionModel(input_data, pre_trained_model, target, pam, num_plot, plot=False):
    profile, rep_reads, in_frame = predictMutations(input_data, pre_trained_model, target, pam)
    sorted_profile = dict(sorted(profile.items(), key=lambda kv: kv[1], reverse=True))
    sorted_profile = {k: v for k, v in sorted_profile.items() if k != '-'}

    profile_freqs = list(profile.values())
    profile_freqs.sort(reverse=True)
    profile_freqs = profile_freqs[1:]

    return sorted_profile


def generate_boxplot():
    with open(f'{config.path}/kl_divs/kl_divs_N=1e03.pkl', 'rb') as f:
        kl_divs_forecast = pkl.load(f)
    f.close()

    with open(f'{config.path}/kl_divs/kl_divs_baseline_N=1e03.pkl', 'rb') as f:
        kl_divs_baseline = pkl.load(f)
    f.close()

    with open(f'C:/Users/Aaron/Desktop/Nanobiology/MSc/MEP/interpreting-ml-based-drops/Lindel/kl_divs/kl_divs_N=1e03_1.pkl', 'rb') as f:
        kl_divs_lindel = pkl.load(f)
    f.close()

    with open(f'C:/Users/Aaron/Desktop/Nanobiology/MSc/MEP/interpreting-ml-based-drops/Lindel/kl_divs/kl_divs_N=1e03_trained.pkl', 'rb') as f:
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
    tijsterman_oligos = os.listdir(f'{config.path}/train/Tijsterman_Analyser')
    DEFAULT_MODEL = config.DEFAULT_MODEL

    oligo_idx = 0

    data_found = False
    num_samples = 0

    kl_divs = {}
    analyze = True
    baseline = False

    if not analyze:
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
            experimental_distribution = dict(sorted(experimental_distribution.items(), key=lambda x: x[0]))

            if baseline:
                baseline_distribution = pd.read_pickle(f"{config.path}/baseline_distribution.pkl")

                if len(experimental_distribution) > len(baseline_distribution):
                    experiment_values = list(experimental_distribution.values())
                    experiment_keys = list(experimental_distribution.keys())

                    experiment_values = experiment_values[:len(baseline_distribution)]
                    experiment_keys = experiment_keys[:len(baseline_distribution)]
                    experimental_distribution = dict(zip(experiment_keys, experiment_values))
                else:
                    baseline_values = list(baseline_distribution.values())
                    baseline_keys = list(baseline_distribution.keys())

                    baseline_values = baseline_values[:len(experimental_distribution)]
                    baseline_keys = baseline_keys[:len(experimental_distribution)]
                    baseline_distribution = dict(zip(baseline_keys, baseline_values))

                KL_div = symmetricKL(experimental_distribution, baseline_distribution)

            else:
                predicted_distribution = model(feature_data)
                predicted_distribution = dict(sorted(predicted_distribution.items(), key=lambda x: x[0]))

                KL_div = symmetricKL(experimental_distribution, predicted_distribution)

            kl_divs[oligo_name] = KL_div

            num_samples += 1
            pbar.update(1)
            oligo_idx += 1
            data_found = False

        pbar.close()

        with open(f'{config.path}/kl_divs/kl_divs_N={config.performance_samples}.pkl', 'wb') as f:
            pkl.dump(kl_divs, f)
    else:
        if baseline:
            with open(f'{config.path}/kl_divs/kl_divs_baseline_N={config.performance_samples}.pkl', 'rb') as f:
                kl_divs = pkl.load(f)
                kl_divs_list = list(kl_divs.values())
                mean_kl_div = np.mean(kl_divs_list)
                print(mean_kl_div)
        else:
            with open(f'C:/Users/Aaron/Desktop/Nanobiology/MSc/MEP/interpreting-ml-based-drops/Lindel/kl_divs/kl_divs_N=1e03_trained.pkl', 'rb') as f:
                kl_divs = pkl.load(f)
                kl_divs_list = list(kl_divs.values())
                mean_kl_div = np.mean(kl_divs_list)
                print(mean_kl_div)

        generate_boxplot()
