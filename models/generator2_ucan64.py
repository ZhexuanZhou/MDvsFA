import torch
from torch import nn
import datetime
import time
from .adnm import AdaptiveBatchNorm2d




class Generator2_UCAN64(nn.Module):
    def __init__(self, channels=64, use_skip=True, is_anm=False):
        super(Generator2_UCAN64, self).__init__()

        Norm2d = AdaptiveBatchNorm2d if is_anm else nn.BatchNorm2d
        if use_skip:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, dilation=1,padding=1),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3,dilation=2,padding=2),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3, dilation=4, padding=4),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3,  dilation=8,padding=8),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3, dilation=16,padding=16),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3, dilation=32,padding=32),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3, dilation=64,padding=64),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer7 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3, dilation=32,padding=32),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # concat as input
            self.layer8 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=channels,kernel_size=3, dilation=16,padding=16),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer9 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=channels,kernel_size=3, dilation=8,padding=8),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer10 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=channels,kernel_size=3, dilation=4,padding=4),
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer11 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=channels,kernel_size=3, dilation=2,padding=2), 
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer12 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=channels,kernel_size=3, dilation=1,padding=1),          
                Norm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer13 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=1,kernel_size=1, dilation=1)
            )

    def forward(self, x):
        # x => (n, 1, 128, 128)
        layer0 = self.layer0(x)
        # x => (n, 64, 128, 128)
        layer1 = self.layer1(layer0)
        # layer1 => (n, 64, 128, 128)
        layer2 = self.layer2(layer1)
        # layer2 => (n, 64, 128, 128)
        layer3 = self.layer3(layer2)
        # layer3 => (n, 64, 128, 128)
        layer4 = self.layer4(layer3)
        # layer4 => (n, 64, 128, 128)
        layer5 = self.layer5(layer4)
        # layer5 => (n, 64, 128, 128)
        out = self.layer6(layer5)
        # layer6 => (n, 64, 128, 128)
        out = self.layer7(out)
        # layer7 => (n, 64, 128, 128)
        out = torch.cat((layer5, out),1)
        out = self.layer8(out)
        # layer8 => (n, 64, 128, 128)
        out = torch.cat((layer4, out),1)
        out = self.layer9(out)
        # a1 = datetime.datetime.now()
        out = torch.cat((layer3, out),1)
        out = self.layer10(out)
       # layer10 => (n, 64, 128, 128)
        out = torch.cat((layer2, out),1)
        out = self.layer11(out)
        # layer11 => (n, 64, 128, 128)
        out = torch.cat((layer1, out),1)
        out = self.layer12(out)
        # layer12 => (n, 64, 128, 128)
        out = self.layer13(out)
        out = torch.clamp(out, 0.0, 1.0)
        # layer13 => (n, 1, 128, 128)
        return out