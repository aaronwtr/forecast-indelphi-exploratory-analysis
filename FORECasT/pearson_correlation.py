import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import config


PEARSON_PLOT_OUTPUT_DIR = '../pearson_ccs_FORECasT/'
FEATURE_DATA_PREFIX = '_pccs_'
FEATURE_OUTPUT_PREFIX = 'pearson_cc_'


def get_pearson_ccs(feature_names, feature_data, mutation, method):
    pearson_ccs = {}
    feature_names_copy = feature_names.copy()
    for feature_a in feature_names:
        feature_names_copy.remove(feature_a)
        feat_a_data = np.array(feature_data[feature_a].tolist())
        for feature_b in feature_names_copy:
            feat_b_data = np.array(feature_data[feature_b].tolist())
            pearson_cc = np.corrcoef(feat_a_data, feat_b_data)[0, 1]
            pearson_ccs[(feature_a, feature_b)] = pearson_cc

    with open(str(method) + '_pccs' + str(mutation) + '.pkl', 'wb') as f:
        pd.to_pickle(pearson_ccs, f)

    return pearson_ccs


def scatter_plot(feature_a, feature_b, feature_data, pearson_ccs):
    fig = px.scatter(feature_data, x=feature_a, y=feature_b, size_max=20)
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )

    fig.update_traces(textposition='top center')

    fig.update_layout(
        height=800,
        title_text='Pearson Correlation Coefficient is {}'.format(pearson_ccs[(feature_a, feature_b)])
    )

    fig.write_image(str(PEARSON_PLOT_OUTPUT_DIR) + str(FEATURE_OUTPUT_PREFIX) + str(feature_a) + '_' + str(feature_b) + '.png')


def get_significant_correlations(pearson_ccs, lower_threshold=0.75, upper_threshold=1):
    # TODO: Remove upper_threshold. Persist this change if the number of feature correlations does not explode.
    significant_correlations = {}
    for key, value in pearson_ccs.items():
        if lower_threshold < abs(value) < upper_threshold:
            significant_correlations[key] = value

    return significant_correlations


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in tqdm(range(0, df.shape[1])):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


if __name__ == '__main__':
    with open(f'E:/Aaron/Nanobiology/MSc/Year3/MEP/train/Tijsterman_Analyser/Oligo_38', 'rb') as f:
        sample = pd.read_pickle(f)

    f, ax = plt.subplots(figsize=(10, 8))
    corr = sample.corr()
    corr_ = corr.fillna(0)
    corr_.values[np.arange(corr_.shape[0]), np.arange(corr_.shape[0])] = 1.0

    indices = np.arange(0, corr.shape[1])
    np.random.seed(2000)
    sampled_indices = np.random.choice(indices, size=30, replace=False)

    corr = corr_.iloc[sampled_indices, sampled_indices]

    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool),
                square=True, ax=ax, vmin=-1, vmax=1, cmap='bwr', xticklabels=False, yticklabels=False)
    plt.tight_layout()
    plt.show()
