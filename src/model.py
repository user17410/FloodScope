"""Small U-Net for the VV and VH radar bands."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=True),
        )


class SmallUNet(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.width = width
        self.encoder1 = ConvBlock(2, width)
        self.encoder2 = ConvBlock(width, width * 2)
        self.encoder3 = ConvBlock(width * 2, width * 4)
        self.bottom = ConvBlock(width * 4, width * 8)
        self.decoder3 = ConvBlock(width * 12, width * 4)
        self.decoder2 = ConvBlock(width * 6, width * 2)
        self.decoder1 = ConvBlock(width * 3, width)
        self.output = nn.Conv2d(width, 1, 1)

    @staticmethod
    def upsample(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        bottom = self.bottom(F.max_pool2d(e3, 2))
        d3 = self.decoder3(self.upsample(bottom, e3))
        d2 = self.decoder2(self.upsample(d3, e2))
        return self.output(self.decoder1(self.upsample(d2, e1)))[:, 0]


def masked_loss(logits, labels):
    valid = labels >= 0
    targets = labels.clamp_min(0).float()
    weights = valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).sum() / weights.sum().clamp_min(1)
    probabilities = logits.sigmoid() * weights
    intersection = (probabilities * targets).sum()
    dice = (2 * intersection + 1) / (probabilities.sum() + (targets * weights).sum() + 1)
    return bce + (1 - dice)
