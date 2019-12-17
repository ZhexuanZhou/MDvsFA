import torch 
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, channels=24):
        super(Discriminator, self).__init__()
        
        # sub-network I
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
            
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # self.nor = nn.Sequential(
        #     nn.BatchNorm2d(1),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        
        self.linear1 = nn.Sequential(
            nn.Linear(1024, 128),
            # nn.BatchNorm1d(128),
            nn.Tanh()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.Tanh()
        )

        self.linear3 = nn.Sequential(
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # x => (n, 2, 128, 128)
        x = self.max_pool1(x)
        # x => (n, 2, 64, 64)
        x = self.max_pool2(x)
        # x => (n, 2, 32, 32)
        x = self.conv1(x)
        # x => (n, 24, 32, 32)
        x = self.conv2(x)
        # x => (n, 24, 32, 32)
        x = self.conv3(x)
        # x => (n, 24, 32, 32)
        x = self.conv4(x)
        # feature_maps = nn.LeakyReLU(0.2, inplace=True)(x)
        # x = self.nor(x)
        feature_maps = x
        # feature_maps => (n, 1, 32, 32)
        x = x.view(-1, 1*32*32)
        # print(x.size())
        # x => (n, 1024, 1, 1)
        x = self.linear1(x)
        # x => (n, 128, 1, 1)
        x = self.linear2(x)
        # x => (n, 64)
        x = self.linear3(x)
        x = nn.Softmax(dim=1)(x)
        # output => (n, 3)
        return x, feature_maps