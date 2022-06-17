from model import LogisticRegression
import torch
import config
from data_preprocessing import get_train_data, get_test_data
import pandas as pd
import pickle as pkl
import Lindel
import os
from tqdm import tqdm
import numpy as np


def accuracy(y_pred, y_true, labels):
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()
    y_pred = pd.DataFrame(y_pred, columns=labels)
    y_true = pd.DataFrame(y_true, columns=labels)
    y_pred = y_pred.sample(n=500)
    y_true = y_true.sample(n=500)
    accs = []

    for (j, row), (k, row_) in zip(y_pred.iterrows(), y_true.iterrows()):
        y_pred = row.to_dict()
        y_true = row_.to_dict()
        y_pred = dict(sorted(y_pred.items(), key=lambda x: x[1], reverse=True))
        y_true = dict(sorted(y_true.items(), key=lambda x: x[1], reverse=True))
        top_y_pred = list(y_pred.keys())[0]
        top_y_true = list(y_true.keys())[0]

        if top_y_pred[0] == top_y_true[0]:
            y_pred_r = top_y_pred.split('R')[1]
            y_true_r = top_y_true.split('R')[1]
            y_pred_len = top_y_pred.split('_')[0][1:]
            y_true_len = top_y_true.split('_')[0][1:]

            y_pred_l = abs(int(y_pred_len) - int(y_pred_r))
            y_true_l = abs(int(y_true_len) - int(y_true_r))

            y_pred_loc = np.zeros(65)
            y_true_loc = np.zeros(65)

            for i_ in range(int(y_pred_r)):
                y_pred_loc[i_ + 30] = 1
            for i_ in range(int(y_true_r)):
                y_true_loc[i_ + 30] = 1

            for i_ in range(int(y_pred_l)):
                y_pred_loc[29 - i_] = 1
            for i_ in range(int(y_true_l)):
                y_true_loc[29 - i_] = 1

            overlap = np.sum(y_pred_loc * y_true_loc)
            total = np.sum(y_true_loc)
            accs.append(overlap / total)

    acc = np.mean(accs)
    return acc


if __name__ == '__main__':
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train = get_train_data(guideset, prerequesites)

    y_labels = y_train.columns.values
    x_train = x_train.values
    x_train = torch.tensor(x_train, dtype=torch.float)
    x_train = x_train.to(device)
    y_train = y_train.values
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_train = y_train.to(device)
    train_size = y_train.shape[1]

    x_test, y_test = get_test_data(guideset, prerequesites, train_size)
    test_data_cols = y_test.columns

    x_test = x_test.values
    x_test = torch.tensor(x_test, dtype=torch.float)
    x_test = x_test.to(device)
    y_test = y_test.values
    y_test = torch.tensor(y_test, dtype=torch.float)
    y_test = y_test.to(device)

    model = LogisticRegression(x_train.shape[1], y_train.shape[1])  # number of features, number of output classes
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_history = []
    test_loss_history = []
    accuracy_history = []
    err_increase = 0

    for epoch in tqdm(range(config.epochs)):
        perm = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], config.batch_size):
            optimizer.zero_grad()
            x_batch = x_train[perm[i:i + config.batch_size]]
            y_batch = y_train[perm[i:i + config.batch_size]]
            y_pred_train = model(x_batch)
            train_loss = criterion(y_pred_train, y_batch)
            train_loss.backward()
            optimizer.step()
        train_loss_history.append(train_loss.item())
        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test)
            accuracy_history.append(accuracy(y_pred_test, y_test, y_labels))
            test_loss = criterion(y_pred_test, y_test)
            test_loss_history.append(test_loss.item())

        print(f'Epoch: {epoch + 1}, Train loss: {train_loss.item():.4f}, Test loss: {test_loss.item():.4f}, Accuracy: '
              f'{accuracy_history[-1]:.4f}')

        if epoch > 2:
            if test_loss_history[-1] > test_loss_history[-2]:
                err_increase += 1

        early_stop_check_sum = np.round(sum(test_loss_history[-config.patience:]), 4)
        early_stop_check_prod = np.round(config.patience * test_loss_history[-1], 4)

        if early_stop_check_sum == early_stop_check_prod or err_increase == config.patience:
            break

    with open(f'{config.path}/losses/train_losses/train_loss_{epoch}_epochs_{config.l2}_weight_decay_{config.lr}_'
              f'learning_rate_{config.batch_size}_batch_size.pkl', 'wb') as file:
        pkl.dump(train_loss_history, file)
    file.close()

    with open(f'{config.path}/losses/test_losses/test_loss_{epoch}_epochs_{config.l2}_weight_decay_{config.lr}_'
              f'learning_rate_{config.batch_size}_batch_size.pkl', 'wb') as file:
        pkl.dump(test_loss_history, file)
    file.close()

    with open(f'{config.path}/losses/accuracies/accuracy_{epoch}_epochs_{config.l2}_weight_decay_{config.lr}_'
              f'learning_rate_{config.batch_size}_batch_size.pkl', 'wb') as file:
        pkl.dump(accuracy_history, file)
    file.close()

    torch.save(model.state_dict(),
               f'{config.path}/model_params/model_params_{epoch}_epochs_{config.l2}_weight_decay.pkl')
