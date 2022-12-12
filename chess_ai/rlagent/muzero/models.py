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

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 1, 9, 8)  # batch_size x channels x board_x x board_y
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
        self.fc1 = nn.Linear(9*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(9*8*128, 4208)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 9*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 9*8*128)
        p = self.fc(p)
        p = F.softmax(p, dim=1)
        return v, p
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for i in range(10):
            setattr(self, f"res_{i}",ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for i in range(10):
            s = getattr(self, f"res_{i}")(s)
        s = self.outblock(s)
        return s
