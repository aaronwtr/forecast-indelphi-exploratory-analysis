import pandas as pd
import os
import pearson_correlation as pearson


def indelphi_pcc():
    data = pd.read_pickle('inDelphi/test_inDelphi.pkl')

    mutation = 'del_features'
    features = pd.DataFrame(data[mutation])
    features_cols = features.columns.tolist()

    if os.path.exists('indelphi_pccs_' + str(mutation) + '.pkl'):
        pearson_ccs = pd.read_pickle('indelphi_pccs_' + str(mutation) + '.pkl')
    else:
        pearson_ccs = pearson.get_pearson_ccs(features_cols, features, mutation, 'indelphi')

    significant_correlations = pearson.get_significant_correlations(pearson_ccs)

    # for corr in significant_correlations:
    #     if str(pearson.FEATURE_OUTPUT_PREFIX) + str(corr[0]) + '_' + str(corr[1]) + '.png' not in os.listdir(pearson.PEARSON_PLOT_OUTPUT_DIR):
    #         pearson.scatter_plot(corr[0], corr[1], features, pearson_ccs)


def forecast_pcc():
    data = pd.read_pickle('FORECasT/train/Tijsterman_Analyser/Oligo_38')
    float_data = data.drop(labels=['Indel'], axis=1)
    feature_cols = float_data.columns.tolist()

    if os.path.exists('forecast_pccs.pkl'):
        pearson_ccs = pd.read_pickle('forecast_pccs.pkl')
    else:
        pearson_ccs = pearson.get_pearson_ccs(feature_cols, data, '', 'forecast')

# TO-DO: Make heatmap of pearson correlation coefficients
def main():
    """
    Note that the features for inDelphi are size, homologyLength and homologyGCContent.
    """

    indelphi = False
    forecast = True

    if indelphi:
        indelphi_pcc()

    if forecast:
        forecast_pcc()


if __name__ == '__main__':
    main()
