import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation Block for 3D data
    Channel-only attention mechanism
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ChannelAttention3D(nn.Module):
    """
    Channel Attention module for CBAM-3D
    Uses both average and max pooling
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention3D(nn.Module):
    """
    Spatial Attention module for CBAM-3D
    Uses channel-wise average and max pooling
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM3D(nn.Module):
    """
    Convolutional Block Attention Module for 3D data
    Combines Channel and Spatial attention
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
