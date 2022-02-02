import pandas as pd
import os
import pearson_correlation as pearson
import numpy as np
import plotly.express as px

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

    pearson_ccs = {k: v for k, v in significant_correlations.items() if not pd.isna(v)}

    return pearson_ccs


def forecast_pcc():
    data = pd.read_pickle('FORECasT/train/Tijsterman_Analyser/Oligo_38')
    float_data = data.drop(labels=['Indel'], axis=1)
    feature_cols = float_data.columns.tolist()

    if os.path.exists('forecast_pccs.pkl'):
        pearson_ccs = pd.read_pickle('forecast_pccs.pkl')
    else:
        pearson_ccs = pearson.get_pearson_ccs(feature_cols, data, '', 'forecast')

    significant_correlations = pearson.get_significant_correlations(pearson_ccs)

    pearson_ccs = {k: v for k, v in significant_correlations.items() if not pd.isna(v)}

    return pearson_ccs


def heatmap(pccs, model):
    feature_keys = list(pccs.keys())
    feature_1 = [feature_keys[i][0] for i in range(len(feature_keys))]
    feature_2 = [feature_keys[i][1] for i in range(len(feature_keys))]
    feature_pccs = list(pccs.values())

    df = pd.DataFrame(np.zeros((len(feature_1), len(feature_2))), index=feature_1, columns=feature_2).astype('float32')
    long_df = pd.DataFrame({'Feature 1': feature_1, 'Feature 2': feature_2, 'Pearson Correlation Coefficent': feature_pccs})
    df2 = long_df.pivot('Feature 1', 'Feature 2', 'Pearson Correlation Coefficent')
    df.add(df2, fill_value=0).add(df2.T, fill_value=0)

    fig = px.imshow(df, color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_layout(title='Pearson Correlation Heatmap of %s' % model)
    fig.show()


def main():
    """
    Note that the features for inDelphi are size, homologyLength and homologyGCContent.
    """

    indelphi = False
    forecast = True

    if indelphi:
        indelphi_pccs = indelphi_pcc()
        heatmap(indelphi_pccs, 'inDelphi')

    if forecast:
        forecast_pccs = forecast_pcc()
        heatmap(forecast_pccs, 'FORECasT')


if __name__ == '__main__':
    main()
