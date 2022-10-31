import pickle
import torch
import pandas as pd
import numpy as np
from chess_ai.rlagent.muzero.models import AlphazeroNet
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
EPOCHS = 500
BATCH_SIZE = 500
DEBUG = True

class BufferDataset(Dataset):
    def __init__(self, x, y_value, y_policy):
        super(BufferDataset, self).__init__()
        assert x.shape[0] == y_value.shape[0] == y_policy.shape[0]
        self.x = x
        self.y_value = y_value
        self.y_policy = y_policy


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y_value[index], self.y_policy[index]
def process(buffer: pd.DataFrame):
    # ouch
    X = torch.tensor(buffer["state"].apply(lambda x: np.array(eval(x))), dtype=torch.float32)
    y_values = torch.tensor(buffer["value"],dtype=torch.float32)
    y_policy = torch.tensor(buffer["policy"].apply(lambda x: np.array(eval(x))), dtype=torch.float32)
    return X, y_values, y_policy

def loss_policy_f(inputs, targets):
        return -torch.sum(targets * inputs) / targets.size()[0]

buffer = pd.read_csv("buffer.csv")

if DEBUG:
    with open("training.pickle", "rb") as f:
        x, y_value, y_policy  = pickle.load(f)
    
else:
    x, y_value, y_policy = process(buffer)
    with open("training.pickle", "wb") as f:
        var = (x,y_value,y_policy)
        pickle.dump(var, f)
dataset = BufferDataset(x=x,y_value=y_value, y_policy=y_policy)
train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=BATCH_SIZE)
model = AlphazeroNet()
loss_v_f = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
model.train()
loss_list = []

for it in range(EPOCHS):
    #list_3dim = []
    for batch, (x, y_value, y_policy) in enumerate(train_dataloader):
        #X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_value_pred, y_policy_pred = model(x)
        #list_3dim.append(model.encoder(model.flatten(X)))        
        loss_value = loss_v_f(y_value_pred, y_value)
        loss_policy = loss_policy_f(y_policy_pred, y_policy)
        loss = loss_value+loss_policy
        loss.backward()
        optimizer.step()
    #print(y_value_pred.mean(), y_value.max())

    loss_list.append(loss.mean().detach().numpy())
    print(it, loss.mean())
torch.save(model.state_dict(), "nn.pth")
plt.plot(loss_list)
plt.show()
        