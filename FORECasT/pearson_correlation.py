import numpy as np
import plotly.express as px
import pandas as pd


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


def get_significant_correlations(pearson_ccs, lower_threshold=0.75, upper_threshold=0.95):
    # TODO: Remove upper_threshold. Persist this change if the number of feature correlations does not explode.
    significant_correlations = {}
    for key, value in pearson_ccs.items():
        if lower_threshold < abs(value) < upper_threshold:
            significant_correlations[key] = value

    return significant_correlations
