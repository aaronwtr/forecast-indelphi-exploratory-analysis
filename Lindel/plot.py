import matplotlib.pyplot as plt
import pandas as pd
import config

# load pkl file
train_loss_1 = pd.read_pickle(f'{config.path}/losses/train_loss_1000_epochs_1e-05_weight_decay.pkl')
train_loss_2 = pd.read_pickle(f'{config.path}/losses/train_loss_1000_epochs_0.0001_weight_decay.pkl')
train_loss_3 = pd.read_pickle(f'{config.path}/losses/train_loss_1000_epochs_0.001_weight_decay.pkl')


# plot training loss
plt.plot(train_loss_1, label='1e-05')
plt.plot(train_loss_2, label='1e-04')
plt.plot(train_loss_3, label='1e-03')
plt.legend()
plt.title('Train loss with varying weight decay')
plt.xlabel('Epoch')
plt.ylabel('Categorical cross entropy loss')
plt.show()
