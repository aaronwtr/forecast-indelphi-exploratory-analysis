import pickle as pkl
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import config
from Lindel.Predictor import *
from get_shap_values import check_pam
from selftarget.view import plotProfiles


def profilePlotter(profile, rep_reads, pam, oligo_idx, plot=True):
    profile_freqs = list(profile.values())
    profile_freqs.sort(reverse=True)

    if plot:
        pp = plotProfiles([profile], [rep_reads], [pam], [False], ['Predicted'], oligo_idx,
                          title='Oligo ' + str(oligo_idx))
        if pp == 0:
            return 0

        plt.savefig(
            f"repair_outcomes/candidate_repair_outcomes/deletions/'Oligo_{oligo_idx}_{config.repair_outcome_of_interest}.pdf")
        plt.close()
        plot = False

        return 1

    return 0


def predict_all_samples(x, guideset, train, truth):
    x_indices = test_data.index.tolist()
    truth_columns = truth.columns.tolist()
    truth = truth.values
    truth = np.log(truth / (1 - truth))
    truth[truth == -np.inf] = 0
    truth = pd.DataFrame(truth, columns=truth_columns)
    # TODO Now the logits can be learned through multinomial, multilabel linear regression. Do not forget to add the
    # TODO softmax step after prediction to return to probabilities.
    models = os.listdir('model_params')

    if 'sklearn_lin_model.pkl' in models:
        with open('model_params/sklearn_lin_model.pkl', 'rb') as f:
            model = pkl.load(f)
    else:
        model = MultiOutputRegressor(estimator=LinearRegression()).fit(train, truth)
        pkl.dump(model, open(f'{config.path}/model_params/sklearn_lin_model', 'wb'))

    if not os.path.exists(f'{config.path}/predicted_repair_outcomes_test_data_lindel.pkl'):
        y_hat = model.predict(x)
        y_hat = y_hat * 5
        y_hat = np.around(y_hat, decimals=3)
        y_hat = np.exp(y_hat) / np.sum(np.exp(y_hat), axis=1).reshape(-1, 1)
        pkl.dump(y_hat, open(f'{config.path}/predicted_repair_outcomes_test_data_lindel.pkl', 'wb'))
    else:
        y_hat = pkl.load(open(f'{config.path}/predicted_repair_outcomes_test_data_lindel.pkl', 'rb'))

    y_hat = pd.DataFrame(y_hat, columns=truth_columns, index=x_indices)

    return y_hat


if __name__ == '__main__':
    pre_trained_weights = pkl.load(open(os.path.join(Lindel.__path__[0], "Model_weights.pkl"), 'rb'))
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    test_data, ground_truth_test = pkl.load(open(f'{config.path}/test_data.pkl', 'rb'))
    train_data, ground_truth_train = pkl.load(open(f'{config.path}/training_data.pkl', 'rb'))

    prediction = predict_all_samples(test_data, guideset, train_data, ground_truth_train)

    ioi = 1796
    test_data_indices = test_data.index.tolist()
    print(test_data_indices[ioi])
    row = prediction.iloc[ioi, :]
    row = row.sort_values(ascending=False)
    print(row)
