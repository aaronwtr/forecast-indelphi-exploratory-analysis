import pandas as pd
import numpy as np
import plotly.express as px
import os


def get_pearson_ccs(feature_names, feature_data):
    pearson_ccs = {}
    feature_names_copy = feature_names.copy()
    for feature_a in feature_names:
        feature_names_copy.remove(feature_a)
        feat_a_data = np.array(feature_data[feature_a].tolist())
        for feature_b in feature_names_copy:
            feat_b_data = np.array(feature_data[feature_b].tolist())
            pearson_cc = np.corrcoef(feat_a_data, feat_b_data)[0, 1]
            pearson_ccs[(feature_a, feature_b)] = pearson_cc

    with open('pearson_ccs.pkl', 'wb') as f:
        pd.to_pickle(pearson_ccs, f)

    return pearson_ccs


def scatter_plot(feature_a, feature_b, feature_data, pearson_ccs):
    fig = px.scatter(feature_data, x=feature_a, y=feature_b, size_max=20)
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    fig.show()

    # find out how to add annotation with pearson cc.


def main():
    data = pd.read_pickle('inDelphi/test_FORECasT_inDelphi.pkl')

    del_features = pd.DataFrame(data["del_features"])
    del_features_cols = del_features.columns.tolist()

    if os.path.exists('pearson_ccs.pkl'):
        pearson_ccs = pd.read_pickle('pearson_ccs.pkl')
    else:
        pearson_ccs = get_pearson_ccs(del_features_cols, del_features)

    scatter_plot(del_features_cols[1], del_features_cols[2], del_features, pearson_ccs)

if __name__ == '__main__':
    main()
