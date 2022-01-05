import pandas as pd
import numpy as np
import plotly.express as px
import os


def get_pearson_ccs(feature_names, feature_data, mutation):
    pearson_ccs = {}
    feature_names_copy = feature_names.copy()
    for feature_a in feature_names:
        feature_names_copy.remove(feature_a)
        feat_a_data = np.array(feature_data[feature_a].tolist())
        for feature_b in feature_names_copy:
            feat_b_data = np.array(feature_data[feature_b].tolist())
            pearson_cc = np.corrcoef(feat_a_data, feat_b_data)[0, 1]
            pearson_ccs[(feature_a, feature_b)] = pearson_cc

    with open('pearson_ccs_' + str(mutation) + '.pkl', 'wb') as f:
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
        title_text='Pearson Correlation Coefficient is {}'.format(pearson_ccs[(feature_a, feature_b)]),
    )

    fig.show()


def get_significant_correlations(pearson_ccs, threshold=0.5):
    significant_correlations = {}
    for key, value in pearson_ccs.items():
        if abs(value) > threshold:
            significant_correlations[key] = value

    return significant_correlations


def main():
    data = pd.read_pickle('inDelphi/test_FORECasT_inDelphi.pkl')

    mutation = 'del_features'
    features = pd.DataFrame(data[mutation])
    features_cols = features.columns.tolist()

    if os.path.exists('pearson_ccs_' + str(mutation) + '.pkl'):
        pearson_ccs = pd.read_pickle('pearson_ccs_' + str(mutation) + '.pkl')
    else:
        pearson_ccs = get_pearson_ccs(features_cols, features, mutation)

    significant_correlations = get_significant_correlations(pearson_ccs)

    print(significant_correlations)

    scatter_plot('Start', 'leftEdge', features, pearson_ccs)


if __name__ == '__main__':
    main()
