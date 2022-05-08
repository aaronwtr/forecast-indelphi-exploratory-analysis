import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    with open('train/Tijsterman_Analyser/Oligo_38', 'rb') as f:
        sample = pd.read_pickle(f)

    f, ax = plt.subplots(figsize=(10, 8))
    corr = sample.corr()
    corr = corr.iloc[3300:3330, 3300:3330]
    # replace nan with 0
    corr = corr.fillna(0)

    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool),
                square=True, ax=ax, vmin=-1, vmax=1, cmap='bwr', xticklabels=False, yticklabels=False)
    # remove x and y labels

    plt.tight_layout()
    plt.show()

    # top_corrs = get_top_abs_correlations(corr, n=20)

    # with open('top_corrs.pkl', 'wb') as f:
    #     pd.to_pickle(top_corrs, f)

    # open top_corrs from file
    # with open('top_corrs.pkl', 'rb') as f:
    #     top_corrs = pd.read_pickle(f)
    #
    # top_corrs_idx = list(top_corrs.index)
    # top_corrs_vals = list(top_corrs.values)
    # top_corr_diag = np.diag(top_corrs_vals)
    # print(top_corr_diag)
    # corr_idx_1 = []
    # corr_idx_2 = []
    # for corr in top_corrs_idx:
    #     corr_idx_1.append(corr[0])
    #     corr_idx_2.append(corr[1])

    # # make a diagonal dataframe with corr_idx_1 and corr_idx_2 as indices and columns respectively and top_corrs_vals as values
    # corr_df = pd.DataFrame(top_corr_diag, index=corr_idx_1, columns=corr_idx_2)
    # top_10 features =
    #
    # # find top 20 pairs of correlated features by correlation coefficient and put them in a square dataframe
    #


