import matplotlib.pyplot as plt
import pandas as pd
import config

# load pkl file
train_loss = pd.read_pickle(f'{config.path}/train_loss.pkl')
print(train_loss)

# plot training loss
plt.plot(train_loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
