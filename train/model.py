import torch.nn as nn


class SRNet(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, scale * scale, 3, padding=1)
        self.relu = nn.ReLU()
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return self.shuffle(x)
