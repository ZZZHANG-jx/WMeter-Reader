from math import log
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import functools
# from cbam import CBAM
import torch.nn.functional as F
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.single_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.Double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            SingleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    def forward(self,x):
        return self.maxpool_conv(x)


class cnn_extractor(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self):
        super().__init__()
        self.down1 = Down(3,16)  ## 24x24
        self.down2 = Down(16,32)  ## 12x12
        self.down3 = Down(32,64)  ## 6x6
        self.down4 = Down(64,128)  ## 3x3
        self.pool = nn.MaxPool2d(3)

    def forward(self, x):
        b,l,c,h,w = x.shape[:]
        x = x.view(b*l,c,h,w)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.pool(x)
        x = x.view(b,l,128).permute(1,0,2)
        return x