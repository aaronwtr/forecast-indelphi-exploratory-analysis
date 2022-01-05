import pandas as pd
import os
import pearson_correlation as pearson


def main():
    """
    Note that the features for inDelphi are size, homologyLength and homologyGCContent.
    """
    data = pd.read_pickle('inDelphi/test_FORECasT_inDelphi.pkl')

    mutation = 'del_features'
    features = pd.DataFrame(data[mutation])
    features_cols = features.columns.tolist()

    if os.path.exists('pearson_ccs_' + str(mutation) + '.pkl'):
        pearson_ccs = pd.read_pickle('pearson_ccs_' + str(mutation) + '.pkl')
    else:
        pearson_ccs = pearson.get_pearson_ccs(features_cols, features, mutation)

    significant_correlations = pearson.get_significant_correlations(pearson_ccs)

    for corr in significant_correlations:
        if str(pearson.FEATURE_OUTPUT_PREFIX) + str(corr[0]) + '_' + str(corr[1]) + '.png' not in os.listdir(pearson.PEARSON_PLOT_OUTPUT_DIR):
            pearson.scatter_plot(corr[0], corr[1], features, pearson_ccs)


if __name__ == '__main__':
    main()
