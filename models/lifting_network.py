import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

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
        residual = x
        out = self.block1(x)
        out = self.block2(out)
        return out + residual
    
class MartinezNet(nn.Module):
    def __init__(self, num_joints_in=28, num_joints_out=28, dropout=0.5):
        super(MartinezNet, self).__init__()
        self.input_proj = LinearBlock(num_joints_in * 2, 1024, dropout)
        self.res1 = ResidualBlock(1024, dropout)
        self.res2 = ResidualBlock(1024, dropout)
        self.output_proj = nn.Linear(1024, num_joints_out * 3)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.output_proj(x)
        return x.view(x.shape[0], -1, 3)
    
if __name__ == "__main__":
    model = MartinezNet()
    dummy = torch.randn(64, 32)
    out = model(dummy)
    print(out.shape)  # should print torch.Size([64, 16, 3])