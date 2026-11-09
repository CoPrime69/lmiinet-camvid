"""U-Net baseline (Ronneberger et al., 2015) with a depth matched to LMIINet.

LMIINet bottoms out at 1/8 resolution, so this U-Net uses 3 down-sampling
steps (4 resolution levels) instead of the original 4. Two widths are used:
  * ``unet``    base=32 -> ~1.9M params (standard U-Net of similar depth)
  * ``unet_pm`` base=20 -> ~0.75M params (parameter-matched to LMIINet)
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = DoubleConv(cout * 2, cout)

    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dh or dw:  # pad for odd sizes
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, num_classes: int = 11, base: int = 32, depth: int = 3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.inc = DoubleConv(3, chs[0])
        self.downs = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[i], chs[i + 1])) for i in range(depth)])
        self.ups = nn.ModuleList([Up(chs[i + 1], chs[i]) for i in reversed(range(depth))])
        self.outc = nn.Conv2d(chs[0], num_classes, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        y = feats.pop()
        for up in self.ups:
            y = up(y, feats.pop())
        return self.outc(y)
