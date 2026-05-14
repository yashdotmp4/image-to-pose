import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(LinearBlock, self).__init__()
        self.linear  = nn.Linear(in_features, out_features)
        self.bn      = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, size, dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.block1 = LinearBlock(size, size, dropout)
        self.block2 = LinearBlock(size, size, dropout)

    def forward(self, x):
        return self.block1(x) + x if False else self.block2(self.block1(x)) + x

class MartinezNet(nn.Module):
    def __init__(self, num_joints_in=17, num_joints_out=17, dropout=0.5, hidden_size=1024):
        super(MartinezNet, self).__init__()
        self.input_proj  = LinearBlock(num_joints_in * 2, hidden_size, dropout)
        self.res1        = ResidualBlock(hidden_size, dropout)
        self.res2        = ResidualBlock(hidden_size, dropout)
        self.output_proj = nn.Linear(hidden_size, num_joints_out * 3)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.output_proj(x)
        return x.view(x.shape[0], -1, 3)