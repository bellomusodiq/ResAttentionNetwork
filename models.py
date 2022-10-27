import torch
import torch.nn as nn
import torchvision

from .layers import AttentionModule


class ResAttentionNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(ResAttentionNet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.attention_module1 = AttentionModule(32, 64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.attention_module2 = AttentionModule(64, 128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.attention_module3 = AttentionModule(128, 256)
        self.avg_pool = nn.AvgPool2d(4)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(256, classes)

    def forward(self, x):
        if x.size(2) > 2:
            x = torchvision.ops.drop_block2d(x, 0.5, 3)
        x = self.conv1(x)
        if x.size(2) > 2:
            x = torchvision.ops.drop_block2d(x, 0.5, 3)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.attention_module1(x)
        x = self.maxpool2(x)
        x = self.attention_module2(x)
        x = self.maxpool3(x)
        x = self.attention_module3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
