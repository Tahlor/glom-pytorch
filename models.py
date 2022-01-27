import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange

def normalize(s):
    for m in s.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return s


class VGG(nn.Module):
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """

    def __init__(self, num_classes=27, pool="max", dropout=True, **kwargs):#62):
        super().__init__()
        self.pool = nn.MaxPool2d if "max" in pool.lower() else nn.AvgPool2d
        self.dropout = nn.Dropout(p=0.5) if dropout else torch.nn.Identity()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(512, num_classes),
        )

    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            self.pool(kernel_size=2, stride=2),
        )
        return normalize(s)

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            self.pool(kernel_size=2, stride=2),
        )
        return normalize(s)


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class VGGLinear(nn.Module):
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """
    def __init__(self, num_classes=27, pool="max", dropout=True, **kwargs):#62):
        super().__init__()
        self.pool = nn.MaxPool2d if "max" in pool.lower() else nn.AvgPool2d
        self.dropout = nn.Dropout(p=0.5) if dropout else torch.nn.Identity()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            self.dropout,
            nn.Linear(512, num_classes),
        )

    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            self.pool(kernel_size=2, stride=2),
        )
        return normalize(s)

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            self.pool(kernel_size=2, stride=2),
        )
        return normalize(s)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class V1(nn.Module):
    def __init__(self, num_classes=27, pool="max", dropout=True, **kwargs):#62):
        super().__init__()
        self.pool = nn.MaxPool2d if "max" in pool.lower() else nn.AvgPool2d
        self.dropout = nn.Dropout(p=0.5) if dropout else torch.nn.Identity()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            self.dropout,
            nn.Linear(512, num_classes),
        )

    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            self.pool(kernel_size=2, stride=2),
        )
        return normalize(s)

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            self.pool(kernel_size=2, stride=2),
        )
        return normalize(s)


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class LinearReg(nn.Module):
    two_conv_pool = three_conv_pool = None
    def __init__(self, num_classes=27, dropout=True, **kwargs):#62):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5) if dropout else torch.nn.Identity()
        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(28*28, num_classes),
        )

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.classifier(x.view(b,-1))
        return F.log_softmax(x, dim=1)