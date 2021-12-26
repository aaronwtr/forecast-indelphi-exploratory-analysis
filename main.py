import pandas as pd
import numpy as np
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


def main():
    data = pd.read_pickle('inDelphi/test_FORECasT_inDelphi.pkl')

    del_features = pd.DataFrame(data["del_features"])
    del_features_cols = del_features.columns.tolist()

    if os.path.exists('pearson_ccs.pkl'):
        pearson_ccs = pd.read_pickle('pearson_ccs.pkl')
    else:
        pearson_ccs = get_pearson_ccs(del_features_cols, del_features)


if __name__ == '__main__':
    main()
