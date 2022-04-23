import pickle as pkl
import numpy as np
import shap
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import random
import os
import config


def open_shap_data(path):
    with open('FORECasT/explanation_datasets/dataset_size_1000/Oligo_58_D3_L-4C5R5.pkl', 'rb') as f:
        feature_data = pkl.load(f)

    files = os.listdir(path)
    values = []

    for file in files:
        with open(path + file, 'rb') as f:
            shap_data = np.array(pkl.load(f))
            shap_data[np.isnan(shap_data)] = 0.0

        values.append(shap_data)

    return values, feature_data


def rank_features(x, feature_data):
    ranked_features_idx = np.argsort(np.mean(np.abs(x), axis=0))
    ranked_features_idx = np.flip(ranked_features_idx[-20:])

    feature_labels = np.array(feature_data.columns)[ranked_features_idx]

    x = pd.DataFrame(x[:, ranked_features_idx], columns=feature_labels)

    return x, feature_labels


def pearson_correlation(x1, x2, feature_data):
    x1, feature_labels_1 = rank_features(x1, feature_data)
    x2, feature_labels_2 = rank_features(x2, feature_data)

    corr = x1.corrwith(x2, axis=0, drop=True).round(2)

    return corr


def get_correlations(x, feature_data):
    corrs = []
    combination_tracker = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j and (j, i) not in combination_tracker:
                corrs.append(pearson_correlation(x[i], x[j], feature_data))
                combination_tracker.append((i, j))

    corrs = pd.concat(corrs, axis=1)

    return corrs


def generate_random_colors(n):
    color_list = []
    random.seed(26)
    for i in range(n):
        color_list.append('#%06X' % random.randint(0, 0xFFFFFF))

    return colors


def generate_colors_from_cmap(x):
    c_map = plt.cm.get_cmap('viridis')

    plt.colormaps()
    arr = np.linspace(0, 1, x.shape[1])
    colorlist = []
    for c in arr:
        rgba = c_map(c)
        clr = colors.rgb2hex(rgba)
        colorlist.append(str(clr))

    return colorlist


def plot_pccs(correlations):
    feature_labels = correlations.index
    colormap_list = generate_colors_from_cmap(correlations)

    for i in range(correlations.shape[1]):
        pcc = correlations.iloc[:, i]
        pcc = pcc.to_numpy()
        plt.scatter(pcc, feature_labels, color=colormap_list[i], alpha=0.8, label=f'Shapley value correlations {i + 1}')

    plt.legend(loc='upper left')
    plt.yticks(feature_labels)
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    '''
    Loading data and obtaining correlations for independent Shapley value experiments run on the same data to assess the
    robustness of the Shapley value method. 
    '''

    base_path = f'FORECasT/shap_save_data/shapley_values/global_explanations/I2/n_1000/nsamples={config.nsamples}/Oligo_5772/'
    shap_values, features = open_shap_data(base_path)
    shap_correlations_plot = True

    if shap_correlations_plot:
        pccs = get_correlations(shap_values, features)
        plot_pccs(pccs)

    '''
    Generating summary plots for a single Shapley value experiment.
    '''

    summary_plot = False
    if summary_plot:
        for value_matrix in shap_values:
            shap.summary_plot(value_matrix, features)
