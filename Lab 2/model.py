import torch
import torch.nn.functional as F
import torch.nn as nn
from math import floor

class snoutNet(nn.Module):
    def __init__(self):
        super(snoutNet, self).__init__()
        # Input: 227x227x3

        # Conv1: 227x227x3 -> 57x57x64
        # k = 3x3x3
        # N_o = (N_i - F + 2P)/S + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=4, padding=1)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        # Conv2: 57x57x64 -> 15x15x128
        # k = 3x3x64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=4, padding=1)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        # Conv3: 15x15x128 -> 4x4x256
        # k = 3x3x128
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=4, padding=1)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        # FC1: 4x4x256 -> 1024
        self.fc1 = nn.Linear(4096, 1024)
        self.relu4 = nn.ReLU()

        # FC2: 1024 -> 1024
        self.fc2 = nn.Linear(1024, 1024)
        self.relu5 = nn.ReLU()

        # FC3: 1024 -> 2
        self.fc3 = nn.Linear(1024, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.mp3(x)

        x = x.view(-1, 4096)  # Flatten

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)

        return x
