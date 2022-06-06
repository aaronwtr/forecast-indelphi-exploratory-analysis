import pandas as pd
from tqdm import tqdm
import pickle as pkl
import config
import torch
from model import LogisticRegression
import numpy as np


def softmax(w):
    return np.exp(w) / sum(np.exp(w))


def predict_all_samples(data):
    """
    Call predict_single_sample in a for loop to predict all the instances in the data. Note that we already have our
    features recorded in the explanation dataset. Therefore, we do not need the check whether the target sequence is
    centered around the PAM again.
    """
    _, out_data = pkl.load(open(f'{config.path}/test_data.pkl', 'rb'))
    weights = torch.load(open(f'{config.path}/model_params/model_params_344_epochs_1e-05_weight_decay.pkl', 'rb'))
    pred_freq = {}

    for i in tqdm(range(data.shape[0])):
        cols = list(out_data.columns.values)
        in_features = data.shape[1]
        out_features = out_data.shape[1]
        # check if data is dataframe or numpy array
        if isinstance(data, pd.DataFrame):
            x = data.iloc[i]
            x = x.values
        else:
            x = data[i]

        x = torch.tensor(x, dtype=torch.float64)

        model = LogisticRegression(in_features, out_features)
        model.load_state_dict(weights)

        y_hat = model(x).tolist()
        y_hat = softmax(y_hat)

        for i in range(len(y_hat)):
            if pred_freq.get(cols[i]) is None:
                pred_freq[cols[i]] = [y_hat[i]]
            else:
                pred_freq[cols[i]].append(y_hat[i])

        for key in pred_freq:
            if len(pred_freq[key]) < len(pred_freq[max(pred_freq, key=len)]):
                pred_freq[key].append(0)

    x = pd.DataFrame.from_dict(pred_freq)
    sample_names = list(data.index)
    x.index = sample_names

    return x


if __name__ == '__main__':
    explanation_data = pkl.load(open(f'{config.path}/explanation_datasets/dataset_size_1000/{config.repair_outcome_of_interest}.pkl', 'rb'))
    output = predict_all_samples(explanation_data)
    print(output)
