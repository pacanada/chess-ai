from torch import nn

class AlphazeroNet(nn.Module):
    def __init__(self):
        super(AlphazeroNet, self).__init__()
        self.l1 = nn.Linear(67, 1000)
        self.l2 = nn.Linear(1000, 1)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.l1(x)
        #x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        return x
