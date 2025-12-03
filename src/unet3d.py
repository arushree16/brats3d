import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True)
        ]
        if norm:
            layers.insert(1, nn.BatchNorm3d(out_ch))
        layers += [
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True)
        ]
        if norm:
            layers.insert(-1, nn.BatchNorm3d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # ensure the spatial size matches (in case of odd sizes)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2,
                        diffZ//2, diffZ - diffZ//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, base_filters=32, num_classes=3):
        super().__init__()
        f = base_filters
        # Encoder
        self.inc = ConvBlock(in_channels, f)
        self.down1 = Down(f, f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)

        # Bottleneck
        self.bottleneck = ConvBlock(f*8, f*16)

        # Decoder
        self.up3 = Up(f*16, f*8)
        self.up2 = Up(f*8, f*4)
        self.up1 = Up(f*4, f*2)
        self.up0 = Up(f*2, f)

        # final conv seg head
        self.outc = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, C, D, H, W]
        x1 = self.inc(x)          # f
        x2 = self.down1(x1)       # f*2
        x3 = self.down2(x2)       # f*4
        x4 = self.down3(x3)       # f*8
        x5 = self.bottleneck(x4)  # f*16

        x = self.up3(x5, x4)      # f*8
        x = self.up2(x, x3)       # f*4
        x = self.up1(x, x2)       # f*2
        x = self.up0(x, x1)       # f
        logits = self.outc(x)     # [B, num_classes, D, H, W]
        return logits
