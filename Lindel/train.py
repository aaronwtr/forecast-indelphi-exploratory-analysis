from model import LogisticRegression
import torch
import config
from get_train_data import get_train_data
import pandas as pd
import pickle as pkl
import Lindel
import os

if __name__ == '__main__':
    guideset = pd.read_csv(f"{config.path}/guideset_data.txt", sep='\t')
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))
    x_train, y_train = get_train_data(guideset, prerequesites)
    feature_data = x_train.values
    feature_data = torch.tensor(feature_data, dtype=torch.FloatTensor)
    print(feature_data.shape[1])

    model = LogisticRegression(feature_data.shape[1], y_train.shape[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        y_pred = model(feature)

        loss = criterion(y_pred, label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
