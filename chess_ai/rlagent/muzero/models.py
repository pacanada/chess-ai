from torch import nn
import torch.nn.functional as F
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

class AlphazeroNetSupervisedOld(nn.Module):
    def __init__(self):
        super(AlphazeroNetSupervisedOld, self).__init__()
        self.l1 = nn.Linear(67, 200)
        #self.l1_a = nn.Linear(100, 200)
        self.l2 = nn.Linear(200, 1)
        self.l3 = nn.Linear(200, 4208)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.l1(x)
        #x = self.l1_a(x)
        #x = self.activation(x)
        x_pol = F.relu(self.l3(x))
        x = self.l2(x)
        return self.activation(x), F.log_softmax(x_pol, dim=1)

class AlphazeroNetSupervised(nn.Module):
    def __init__(self):
        super(AlphazeroNetSupervised, self).__init__()
        self.c1 = nn.Conv1d(1, 20, kernel_size=3, stride=1)
        self.l1 = nn.Linear(20*(67-2),1)
        self.l2 = nn.Linear(20*(67-2),4208)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = x.view(-1,1,67)
        x = self.c1(x)
        #x = self.l1_a(x)
        #x = self.activation(x)
        x_pol = F.relu(self.l2(x.view(-1,20*(67-2))))
        x = self.l1(x.view(-1,20*(67-2)))
        return self.activation(x), F.softmax(x_pol, dim=1)
