import matplotlib.pyplot as plt
import os
import pickle as pkl
import config


if __name__ == '__main__':
    train_loss_files = os.listdir('losses/train_losses')
    test_loss_files = os.listdir('losses/test_losses')
    train_loss_files.remove('archive')
    test_loss_files.remove('archive')
    train_losses = []
    test_losses = []

    for file in test_loss_files:
        with open(f'losses/test_losses/{file}', 'rb') as f:
            test_loss = pkl.load(f)
            test_losses.append(test_loss)

    for file in train_loss_files:
        with open(f'losses/train_losses/{file}', 'rb') as f:
            train_loss = pkl.load(f)
            train_losses.append(train_loss)

    weight_decays = [f'{file_name.split("_")[4]}' for file_name in train_loss_files]
    wds = {i: wd for i, wd in enumerate(weight_decays)}
    wds = {k: v for k, v in sorted(wds.items(), key=lambda item: item[1])}
    weight_decays = list(wds.values())
    sorted_indices = list(wds.keys())
    learning_rates = [f'{file_name.split("_")[7]}' for file_name in train_loss_files]
    learning_rates = [learning_rates[i] for i in sorted_indices]
    batch_sizes = [f'{file_name.split("_")[10][:-4]}' for file_name in train_loss_files]
    batch_sizes = [batch_sizes[i] for i in sorted_indices]
    train_losses = [train_losses[i] for i in sorted_indices]

    # for i, loss in enumerate(train_losses):
    #     plt.plot(loss, label=f'', alpha=1/(i/15 + 1))

    plt.plot(train_loss, label='train', alpha=1)
    plt.plot(test_loss, label='test', alpha=1)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Test loss (KL-divergence)')
    plt.show()
