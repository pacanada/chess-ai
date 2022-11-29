from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from chess_ai.rlagent.muzero.models import AlphazeroNetSupervised, AlphazeroNetSupervisedOld

from chess_ai.rlagent.muzero.utils import MOVES, BufferDataset, get_root_dir, process_buffer_to_torch, process_buffer_to_torch_state_64

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 1, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(8*8*128, 4208)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = F.softmax(p, dim=1)
        return v, p
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for i in range(1):
            setattr(self, f"res_{i}",ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for i in range(1):
            s = getattr(self, f"res_{i}")(s)
        s = self.outblock(s)
        return s


batch_size = 100
epochs = 1000
buffer = pd.read_feather(get_root_dir() / "data" / "dataframe" / f"buffer_df.feather").head(100)
print(buffer.shape)
# buffer_1 = pd.read_feather(get_root_dir() / "data" / "dataframe" / "buffer_1_df.feather")
#buffer["value_all"] = buffer.evaluation.apply(lambda x: x["value"] if x["type"]=="cp" else x["value"]*10000)
#buffer["value"] = 1/(1+np.exp(-0.01*buffer.evaluation))
#buffer["mate_value"] = buffer.evaluation.apply(lambda x: x["value"] if x["type"]=="mate" else None)
model = ChessNet()
# model.load_state_dict(torch.load(get_root_dir() / "checkpoints/nn_supervised_res.pth"))
# model.eval()
x, y_value, y_policy = process_buffer_to_torch_state_64(buffer)
print("processed to torch")
dataset = BufferDataset(x=x,y_value=y_value, y_policy=y_policy)
train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

loss_v_f = torch.nn.MSELoss()
loss_policy_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
model.train()

loss_list = []

for it in range(epochs):
    for x, y_value, y_policy in train_dataloader:
        optimizer.zero_grad()
        y_value_pred, y_policy_pred = model(x)        
        loss_value = loss_v_f(y_value_pred, y_value)
        loss_policy = loss_policy_f(y_policy_pred, y_policy)
        loss = 40*loss_value+loss_policy
        #loss = loss_policy
        loss.backward()
        optimizer.step()

    loss_list.append(loss.mean().detach().numpy())
    print(f"Epoch: {it}/{epochs}, loss: {loss.mean()}")
    if it%50==0:
        torch.save(model.state_dict(), get_root_dir() /"checkpoints/nn_supervised_res.pth")
        print("saving")
torch.save(model.state_dict(), get_root_dir() /"checkpoints/nn_supervised_res.pth")
plt.plot(loss_list)
plt.show()