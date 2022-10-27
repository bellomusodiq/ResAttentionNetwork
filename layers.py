import torch
import torch.nn as nn
import torchvision


class ResidualUnit(nn.Module):
    """
    basic convolution unit, this uses pre-activated residual block
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels // 2, kernel_size=3, padding=1, stride=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1)
        self.residual_transform = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1
        )

    def forward(self, x):
        residual = x
        if x.size(2) > 2:
            x = torchvision.ops.drop_block2d(x, 0.5, 3)
        x = self.conv1(self.relu(self.bn1(x)))
        if x.size(2) > 2:
            x = torchvision.ops.drop_block2d(x, 0.5, 3)
        x = self.conv2(self.relu(self.bn2(x)))
        if x.size(2) > 2:
            x = torchvision.ops.drop_block2d(x, 0.5, 3)
        x = self.conv3(self.relu(self.bn3(x)))
        if self.out_channels != self.in_channels:
            residual = self.residual_transform(residual)
        sum = residual + x
        return sum


class SoftMaxBranch(nn.Module):
    """
    soft max branch layer, implemets an encoder-decoder connection, similar to convolution
    auto-encoders, and sigmoid activation function is used to scale output between 0-1
    """

    def __init__(self, in_channels):
        super(SoftMaxBranch, self).__init__()
        self.in_channels = in_channels
        self.maxpool1 = nn.MaxPool2d(2)
        self.down_unit1 = ResidualUnit(in_channels, in_channels)
        self.maxpool2 = nn.MaxPool2d(2)
        self.down_unit2 = ResidualUnit(in_channels, in_channels)
        self.up_unit1 = ResidualUnit(in_channels, in_channels)
        self.interpolation1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up_unit2 = ResidualUnit(in_channels, in_channels)
        self.interpolation2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool1(x)
        x = self.down_unit1(x)
        x = self.maxpool2(x)
        x = self.down_unit2(x)
        x = self.up_unit1(x)
        x = self.interpolation1(x)
        x = self.up_unit2(x)
        x = self.interpolation2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class AttentionModule(nn.Module):
    """
    Attention module combines the trunk branch and mask branch
    """

    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unit1 = ResidualUnit(in_channels, out_channels)
        self.trunk_branch = nn.Sequential(
            ResidualUnit(out_channels, out_channels),
            ResidualUnit(out_channels, out_channels),
        )
        self.mask_branch = SoftMaxBranch(out_channels)
        self.unit2 = ResidualUnit(out_channels, out_channels)

    def forward(self, x):
        x = self.unit1(x)
        trunk = self.trunk_branch(x)
        mask = self.mask_branch(x)
        skip_connection = trunk
        trunk_scale = trunk * mask
        out = skip_connection + trunk_scale
        out = self.unit2(out)
        return out
