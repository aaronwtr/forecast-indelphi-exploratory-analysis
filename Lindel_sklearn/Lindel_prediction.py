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

        plt.savefig(f"repair_outcomes/candidate_repair_outcomes/deletions/'Oligo_{oligo_idx}_{config.repair_outcome_of_interest}.pdf")
        plt.close()
        plot = False

        return 1

    return 0


def predict_all_samples(x, guideset, train, truth):
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
        pkl.dump(model, open(f'model_params/sklearn_lin_model'))

    y_hat = model.predict(x)

    # y_hat = model.predict_proba(x)
    # y_hat = model(x).tolist()
    # y_hat = [round(x * 10, 3) for x in y_hat]
    # y_hat = softmax(y_hat)
    #
    # cols = list(truth.columns.values)
    # pred_freq = {}
    # for i in range(len(y_hat)):
    #     pred_freq[cols[i]] = y_hat[i]
    #
    # pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1], reverse=True)
    # pred_sorted = {k: v for k, v in pred_sorted}
    # pred_sorted = {k: v for k, v in pred_sorted.items() if not k.startswith('Indel_')}
    # print(pred_sorted)

    # tmp_genindels_file = 'tmp_genindels_%s_%d.txt' % (seq, random.randint(0, 100000))
    # cmd = INDELGENTARGET_EXE + ' %s %d %s' % (seq, pam_idx, tmp_genindels_file)
    #
    # subprocess.check_call(cmd.split())
    # rep_reads = fetchRepReads(tmp_genindels_file)
    #
    # profilePlotter(pred_sorted, rep_reads, pam_idx, oligo_name)

    return pred_sorted


if __name__ == '__main__':
    pre_trained_weights = pkl.load(open(os.path.join(Lindel.__path__[0], "Model_weights.pkl"), 'rb'))
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    test_data, ground_truth_test = pkl.load(open(f'{config.path}/test_data.pkl', 'rb'))
    train_data, ground_truth_train = pkl.load(open(f'{config.path}/training_data.pkl', 'rb'))

    prediction = predict_all_samples(test_data, guideset, train_data, ground_truth_train)
