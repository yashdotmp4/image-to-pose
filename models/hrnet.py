import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)
    
class FusionLayer(nn.Module):
    def __init__(self, num_branches, channels):
        super(FusionLayer, self).__init__()
        self.num_branches = num_branches
        self.fuse_layers = nn.ModuleList()
        
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    # upsample from lower to higher resolution
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j < i:
                    # downsample from higher to lower resolution
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            convs.append(nn.Sequential(
                                nn.Conv2d(channels[j], channels[i], 3, 
                                         stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(channels[i])
                            ))
                        else:
                            convs.append(nn.Sequential(
                                nn.Conv2d(channels[j], channels[j], 3, 
                                         stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*convs))
                else:
                    fuse_layer.append(None)
            self.fuse_layers.append(fuse_layer)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = []
        for i in range(self.num_branches):
            result = x[i]
            for j in range(self.num_branches):
                if j != i and self.fuse_layers[i][j] is not None:
                    result = result + self.fuse_layers[i][j](x[j])
            out.append(self.relu(result))
        return out
    
class HRModule(nn.Module):
    def __init__(self, num_branches, channels, num_blocks):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            blocks = []
            for _ in range(num_blocks):
                blocks.append(BasicBlock(channels[i], channels[i]))
            self.branches.append(nn.Sequential(*blocks))
        
        self.fusion = FusionLayer(num_branches, channels)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        return self.fusion(x)

class HRNet(nn.Module):
    def __init__(self, num_keypoints=17, width=48):
        super(HRNet, self).__init__()
        
        # stem - initial feature extraction
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=2, padding=1),
            BottleneckBlock(64, 64),
            BottleneckBlock(256, 64),
            BottleneckBlock(256, 64),
            BottleneckBlock(256, 64),
        )
        
        # transition to stage 2 - create two streams
        self.transition1 = nn.ModuleList([
            ConvBNReLU(256, width, 3, padding=1),
            nn.Sequential(
                ConvBNReLU(256, width*2, 3, stride=2, padding=1)
            )
        ])
        
        # stage 2 - two parallel streams
        self.stage2 = nn.Sequential(
            HRModule(2, [width, width*2], num_blocks=4)
        )
        
        # transition to stage 3 - create third stream
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                ConvBNReLU(width*2, width*4, 3, stride=2, padding=1)
            )
        ])
        
        # stage 3 - three parallel streams
        self.stage3 = nn.Sequential(
            HRModule(3, [width, width*2, width*4], num_blocks=4),
            HRModule(3, [width, width*2, width*4], num_blocks=4),
            HRModule(3, [width, width*2, width*4], num_blocks=4),
            HRModule(3, [width, width*2, width*4], num_blocks=4),
        )
        
        # transition to stage 4 - create fourth stream
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                ConvBNReLU(width*4, width*8, 3, stride=2, padding=1)
            )
        ])
        
        # stage 4 - four parallel streams
        self.stage4 = nn.Sequential(
            HRModule(4, [width, width*2, width*4, width*8], num_blocks=4),
            HRModule(4, [width, width*2, width*4, width*8], num_blocks=4),
            HRModule(4, [width, width*2, width*4, width*8], num_blocks=4),
        )
        
        # final prediction head - only uses highest resolution stream
        self.final_layer = nn.Conv2d(width, num_keypoints, 1)

    def forward(self, x):
        # stem
        x = self.stem(x)
        
        # transition to stage 2
        x = [t(x) for t in self.transition1]
        
        # stage 2
        x = self.stage2[0](x)
        
        # transition to stage 3
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[1])
        ]
        
        # stage 3
        for module in self.stage3:
            x = module(x)
        
        # transition to stage 4
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[2])
        ]
        
        # stage 4
        for module in self.stage4:
            x = module(x)
        
        # only take highest resolution stream
        x = self.final_layer(x[0])
        return x