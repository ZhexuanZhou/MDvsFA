import torch 
from torch import nn

import datetime

from .adnm import AdaptiveBatchNorm2d


class Generator1_CAN8(nn.Module):
    def __init__(self, channels=64, is_anm=False):
        super(Generator1_CAN8, self).__init__()
        Norm2d = AdaptiveBatchNorm2d if is_anm else nn.BatchNorm2d
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, 
                kernel_size=3, dilation=1, padding=1),
            Norm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                kernel_size=3, dilation=1, padding=1),
            Norm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels,
                kernel_size=3, dilation=2, padding=2),
            Norm2d(2*channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=2*channels, out_channels=4*channels,
                kernel_size=3, dilation=4, padding=4),
            Norm2d(4*channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=4*channels, out_channels=8*channels,
                kernel_size=3, dilation=8, padding=8),
            Norm2d(8*channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=8*channels, out_channels=4*channels,
                kernel_size=3, dilation=4, padding=4),
            Norm2d(4*channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=4*channels, out_channels=2*channels,
                kernel_size=3, dilation=2, padding=2),
            Norm2d(2*channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=2*channels, out_channels=channels,
                kernel_size=3, dilation=1, padding=1),
            Norm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1,
                kernel_size=1, dilation=1)
        )
        
        
    def forward(self, x):
        # x => (10, 1, 128, 128)
        # print(x.size())
        x = self.layer0(x)  # x => (10, 64, 128, 128)
        # print(x.size())
        x = self.layer1(x)  # x => (10, 64, 128, 128)
        # print(x.size())
        x = self.layer2(x)  # x => (10, 128, 128, 128)
        # print(x.size())
        x = self.layer3(x)# x => (10, 256, 128, 128)
        # print(x.size())
        x = self.layer4(x)# x => (10, 512, 128, 128)
        # print(x.size())
        x = self.layer5(x)# x => (10, 256, 128, 128)
        # print(x.size())
        x = self.layer6(x)# x => (10, 128, 128, 128)
        # print(x.size())
        x = self.layer7(x)# x => (10, 64, 128, 128)
        # print(x.size())
        x = self.layer8(x)# x => (10, 1, 128, 128)
        # a1 = datetime.datetime.now()
        
        x = torch.clamp(x, 0.0, 1.0)

        return x