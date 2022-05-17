import matplotlib.pyplot as plt
import pandas as pd
import config

# load pkl file
train_loss_1 = pd.read_pickle(f'{config.path}/losses/train_loss_247_epochs_1e-05_weight_decay.pkl')
test_loss_1 = pd.read_pickle(f'{config.path}/losses/test_loss_247_epochs_1e-05_weight_decay.pkl')

# plot training loss
plt.plot(train_loss_1, label='training loss 1e-05')
plt.plot(test_loss_1, label='test loss 1e-05')
# set legend title
plt.legend()
plt.title('Train loss 1e-05 learning rate, 1e-05 weight decay, early stopping patience = 3')
plt.xlabel('Epoch')
plt.ylabel('Categorical cross entropy loss')
plt.show()
