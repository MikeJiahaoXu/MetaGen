import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from PIL import Image
from dataset import MyDataset
from torch.utils.data import DataLoader

def weights_init(m):
    """weights initialization"""
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



class Prediction_Network(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.condition_dim = condition_dim
        # opearation for image, use multiple kernel size
        self.cnn1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # operation for condition
        self.fc1 = nn.Sequential(
            nn.Linear(self.condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4096),
        )
        # operation for combined image and condition
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # output prediction of S-parameters
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 50),
        )
        self.dropout1 = nn.Dropout(p=0.3)
        self.apply(weights_init)

    def forward(self, x, c):
        x1_1 = self.cnn1_1(x)
        x1_2 = self.cnn1_2(x)
        x1_3 = self.cnn1_3(x)
        # concat three different kernel size
        x1 = torch.cat([x1_1, x1_2, x1_3], dim=1)
        x2 = self.cnn2(x1)
        x3 = self.cnn3(x2)
        # operation for condition
        c1 = self.fc1(c)
        # c1 = self.dropout1(c1)
        c1 = c1.view(c1.shape[0], 64, 8, 8)
        # concat image and condition
        cat_tensor = torch.cat((x3, c1), dim=1)
        y = self.cnn4(cat_tensor)
        y = y.view((y.shape[0], -1))
        # y = self.dropout1(y)
        y = self.fc2(y)
        # output prediction of S-parameters
        return y
