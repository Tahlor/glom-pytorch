import torch
import torch.nn.functional as F
from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels=1, f1=32, f2=32):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.net(x)


class GeometricMean(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        log_x = torch.log(F.relu(x))
        return torch.exp(torch.mean(log_x, dim=self.dim))

class Mean(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x,self.dim)

if __name__ == '__main__':
    C = CNN()
    x = torch.randn(2,1,28,28)
    o = C(x)
    print(o.shape)
    pass