import torch.nn as nn

def conv_batchnorm_relu(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class Conv4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            conv_batchnorm_relu(3, 64),
            nn.MaxPool2d(2),
            conv_batchnorm_relu(64, 64),
            nn.MaxPool2d(2),
            conv_batchnorm_relu(64, 64),
            nn.MaxPool2d(2),
            conv_batchnorm_relu(64, 64),
            nn.AdaptiveMaxPool2d((1,1)),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(x.size(0), -1)