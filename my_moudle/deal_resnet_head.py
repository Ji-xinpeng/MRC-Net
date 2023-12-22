import torch
from deals.deal_picture import *
import torch.nn as nn

class DealResnetHead(nn.Module):
    def __init__(self):
        super(DealResnetHead, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """
        x: 进来的时候是 NT, C, H, W , 要先变成 nt, h, w, c
        return: 的时候是 nt, 4*c, h//2, w//2
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        nt, h, w, c = x.shape

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)

        return x


class ReplaceMaxPooling(nn.Module):
    def __init__(self):
        super(ReplaceMaxPooling, self).__init__()
        self.replace_maxpooling = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.replace_maxpooling(x)

        return x

