import torch
import torch.nn as nn
from others.params import *


class MySEBlock(nn.Module):
    def __init__(self, channel):
        super(MySEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // args.reduce_num, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // args.reduce_num, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        nt, c, h, w = x.size()
        y = self.avg_pool(x).view(nt, c)
        y = self.fc(y).view(nt, c, 1, 1)

        return y



# x = torch.randn(16, 64, 56, 56)
# se = MySEBlock(64)
# Y = se(x)
# print(Y.shape)
