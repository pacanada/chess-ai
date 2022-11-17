import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch

from chess_ai.rlagent.muzero.utils import BufferDataset, get_root_dir, process_buffer_to_torch
class AlphazeroNet(nn.Module):
    def __init__(self):
        super(AlphazeroNet, self).__init__()
        self.l1 = nn.Linear(67, 10)
        self.l2 = nn.Linear(10, 1)
        self.l3 = nn.Linear(10, 4208)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.l1(x)
        #x = self.activation(x)
        x_pol = F.relu(self.l3(x))
        x = self.l2(x)
        return self.activation(x), F.softmax(x_pol, dim=1)


batch_size = 1000
epochs = 100
buffer = pd.read_feather("supervised_dataset.feather")
buffer["value_all"] = buffer.evaluation.apply(lambda x: x["value"] if x["type"]=="cp" else x["value"]*10000)
buffer["value"] = 1/(1+np.exp(-0.01*buffer.value_all))
#buffer["mate_value"] = buffer.evaluation.apply(lambda x: x["value"] if x["type"]=="mate" else None)
model = AlphazeroNet()
model.load_state_dict(torch.load(get_root_dir() / "checkpoints/nn_latest.pth"))
model.eval()
x, y_value, y_policy = process_buffer_to_torch(buffer)
dataset = BufferDataset(x=x,y_value=y_value, y_policy=y_policy)
train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

loss_v_f = torch.nn.MSELoss()
loss_policy_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
model.train()

loss_list = []

for it in range(epochs):
    for x, y_value, y_policy in train_dataloader:
        optimizer.zero_grad()
        y_value_pred, y_policy_pred = model(x)        
        loss_value = loss_v_f(y_value_pred, y_value)
        loss_policy = loss_policy_f(y_policy_pred, y_policy)
        loss = loss_value+loss_policy
        loss.backward()
        optimizer.step()

    loss_list.append(loss.mean().detach().numpy())
    print(it, loss.mean())
torch.save(model.state_dict(), get_root_dir() /"checkpoints/nn_supervised.pth")