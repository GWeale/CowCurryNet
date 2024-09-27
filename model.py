import torch
import torch.nn as nn
import torch.nn.functional as F

class attentionlayer(nn.Module):
    def __init__(self, in_channels):
        super(attentionlayer, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attn(x)

class resblock(nn.Module):
    def __init__(self, in_channels):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class customalexnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(customalexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            attentionlayer(96),
            resblock(96),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            attentionlayer(256),
            resblock(256),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            attentionlayer(256),
            resblock(256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(num_classes=1000):
    model = customalexnet(num_classes)
    return model
