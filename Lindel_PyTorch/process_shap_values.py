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
    with open(f'{config.path}/explanation_datasets/dataset_size_1000/Oligo_4698_D18_L-19C12R12.pkl', 'rb') as f:
        feature_data = pkl.load(f)

    files = os.listdir(path)
    values = []

    file = "Oligo_4698_D18_L-19C12R12_global_shap_values_2.pkl"

    with open(path + file, 'rb') as f:
        if config.shap_type == 'global':
            shap_vals = pkl.load(f)
            shap_vals = np.array(shap_vals)
            shap_vals[np.isnan(shap_vals)] = 0.0
            shap_vals = shap_vals[0, :, :]
            print(shap_vals.shape)
        else:
            shap_vals, ex_value = pkl.load(f)
            shap_vals[np.isnan(shap_vals)] = 0.0
            values.append(shap_vals)

            return values, feature_data, ex_value

    values.append(shap_vals)

    return values, feature_data


def rank_features(x, feature_data):
    ranked_features_idx = np.argsort(np.mean(np.abs(x), axis=0))
    ranked_features_idx = ranked_features_idx[-20:]

    feature_labels = np.array(feature_data.columns)[ranked_features_idx]

    x_ranked = x[:, ranked_features_idx]
    x_ranked = np.flip(np.sort(x_ranked, axis=0), axis=0)

    x = pd.DataFrame(x_ranked, columns=feature_labels)

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

    if len(corrs) > 0:
        corrs = pd.concat(corrs, axis=1)
    else:
        corrs = pd.DataFrame(corrs, columns=feature_data.columns)

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

    plt.legend(loc='lower left')
    plt.yticks(feature_labels)
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    '''
    Loading data and obtaining correlations for independent Shapley value experiments run on the same data to assess the
    robustness of the Shapley value method. 
    '''

    base_path = f'{config.lindel_hd_path}/shap_save_data/shapley_values/global_explanations/D18/n_100/nsamples=200/'
    if config.shap_type == 'global':
        shap_values, features = open_shap_data(base_path)
    else:
        shap_values, features, ex_value = open_shap_data(base_path)

    shap_correlations_plot = False

    if shap_correlations_plot:
        pccs = get_correlations(shap_values, features)
        plot_pccs(pccs)

    '''
    Generating summary plots for a single instance in the repair outcome dataset.
    '''

    features = features.iloc[shap_values[0].shape[0], :]
    summary_plot = False
    if summary_plot:
        for value_matrix in shap_values:
            shap.summary_plot(value_matrix, features)

    ''''
    Generating force- and bar plot for a single instance in the repair outcome dataset.
    '''

    force_plot = True
    if force_plot:
        shap.initjs()
        plt.rcParams['ytick.labelsize'] = 'small'
        shap.bar_plot(shap_values[0], max_display=10, show=False)

        ytick_labels_tmp = plt.gca().get_yticklabels()
        ytick_labels = [ytick_label.get_text() for ytick_label in ytick_labels_tmp]
        ytick_labels_idx = [int(ytick_label.split(' ')[1]) for ytick_label in ytick_labels]
        feature_names = features.columns
        feature_names_at_idx = [feature_names[idx] for idx in ytick_labels_idx]
        ytick_locs = [i + 1 for i in range(len(feature_names_at_idx))]
        ytick_locs = ytick_locs[::-1]
        plt.yticks(ytick_locs, feature_names_at_idx)
        plt.tight_layout()
        plt.show()

        force = shap.force_plot(ex_value, shap_values[0], features.iloc[0, :], contribution_threshold=0.5)
        shap.save_html(f'{config.path}/shap_values_visualizations/force_plots/results/D18/n_1000/'
                       f'nsamples={config.nsamples}/{config.indel_of_interest}_force_plot.html', force)
        plt.clf()
        plt.close()
