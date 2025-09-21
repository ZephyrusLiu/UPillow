import torch
import torch.nn as nn
import torch.nn.functional as F

class TinySleepCNNMulti(nn.Module):
    """
    A simple multi-channel 1D CNN.
    Input shape: (B, C, T)
    """
    def __init__(self, n_classes=5, in_len=3000, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 32, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        # compute flat dim
        with torch.no_grad():
            x = torch.zeros(1, in_ch, in_len)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flat = x.numel()
        self.fc1 = nn.Linear(flat, 256)
        self.do = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do(x)
        return self.fc2(x)
