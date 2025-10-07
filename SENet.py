import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 1x1xC
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 1x1xC/r
            nn.ReLU(inplace=True),  # 1x1xC/r
            nn.Linear(channel // reduction, channel, bias=False),  # 1x1xC
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)