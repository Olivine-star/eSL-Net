import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.act(residual)
        residual = self.conv2(residual)
        return self.act(x + residual)


class LowLightEnhanceNet(nn.Module):
    """Small RGB U-Net for paired low-light enhancement."""

    def __init__(self, base_channels=32, residual_blocks=4):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(3, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.bottleneck_in = ConvBlock(c3, c3)
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(c3) for _ in range(residual_blocks)]
        )

        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        mid = self.bottleneck(self.bottleneck_in(self.down2(enc2)))

        up2 = F.interpolate(mid, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        residual = self.out(dec1)
        x_logit = torch.logit(x.clamp(1e-4, 1.0 - 1e-4))
        return torch.sigmoid(x_logit + residual)
