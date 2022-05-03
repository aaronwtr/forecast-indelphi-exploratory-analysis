import matplotlib.pyplot as plt
import pandas as pd
import config

# load pkl file
train_loss_1 = pd.read_pickle(f'{config.path}/losses/train_loss_387_epochs_0.001_weight_decay.pkl')
test_loss_1 = pd.read_pickle(f'{config.path}/losses/test_loss_387_epochs_0.001_weight_decay.pkl')

# plot training loss
plt.plot(train_loss_1, label='training loss')
plt.plot(test_loss_1, label='test loss')
plt.legend()
plt.title('Train loss 1e-3 weight decay, 1e-5 learning rate, patience = 3')
plt.xlabel('Epoch')
plt.ylabel('Categorical cross entropy loss')
plt.show()
