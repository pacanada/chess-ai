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
