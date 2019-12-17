import torch
from torch import nn


class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1).fill_(1.0))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1).fill_(0.0))
        
    def forward(self, x):
        x = self.a * x + self.b * self.bn(x)
        return x