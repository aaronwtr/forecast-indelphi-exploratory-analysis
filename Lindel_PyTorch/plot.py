import matplotlib.pyplot as plt
import os
import pickle as pkl
import config


if __name__ == '__main__':
    train_loss_files = os.listdir('losses/train_losses')
    test_loss_files = os.listdir('losses/test_losses')
    accuracy_files = os.listdir('losses/accuracies')
    train_loss_files.remove('archive')
    test_loss_files.remove('archive')
    train_losses = []
    test_losses = []
    accuracies = []

    for file in test_loss_files:
        with open(f'losses/test_losses/{file}', 'rb') as f:
            test_loss = pkl.load(f)
            test_losses.append(test_loss)

    for file in train_loss_files:
        with open(f'losses/train_losses/{file}', 'rb') as f:
            train_loss = pkl.load(f)
            train_losses.append(train_loss)

    for file in accuracy_files:
        with open(f'losses/accuracies/{file}', 'rb') as f:
            accuracy = pkl.load(f)
            accuracies.append(accuracy)

    weight_decays = [f'{file_name.split("_")[4]}' for file_name in test_loss_files]
    # convert all scientific notation to float
    weight_decays = [float(weight_decay) for weight_decay in weight_decays]
    wds = {i: wd for i, wd in enumerate(weight_decays)}
    wds = {k: v for k, v in sorted(wds.items(), key=lambda item: item[1])}
    weight_decays = list(wds.values())
    sorted_indices = list(wds.keys())
    learning_rates = [f'{file_name.split("_")[7]}' for file_name in test_loss_files]
    learning_rates = [learning_rates[i] for i in sorted_indices]
    batch_sizes = [f'{file_name.split("_")[10]}' for file_name in test_loss_files]
    batch_sizes = [batch_sizes[i] for i in sorted_indices]
    test_losses = [test_losses[i] for i in sorted_indices]
    accs = [accuracies[i] for i in sorted_indices]

    for i, loss in enumerate(test_losses):
        plt.plot(loss, label=f'wd={weight_decays[i]}, lr={learning_rates[i]}, bs={batch_sizes[i]}', alpha=1/(i/15 + 1))

    # plt.plot(test_loss, label='test', alpha=1)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Test loss (KL-divergence)')
    plt.show()

    for i, acc in enumerate(accs):
        plt.plot(acc, label=f'wd={weight_decays[i]}, lr={learning_rates[i]}, bs={batch_sizes[i]}', alpha=1/(i/15 + 1))

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
