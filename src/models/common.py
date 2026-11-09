"""Shared building blocks used by LMIINet (and partly by the baselines)."""
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

IntOrPair = Union[int, Tuple[int, int]]


def _pair(v: IntOrPair) -> Tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm -> (optional) PReLU, with 'same' padding by default."""

    def __init__(self, cin: int, cout: int, k: IntOrPair = 3, s: int = 1,
                 d: IntOrPair = 1, g: int = 1, act: bool = True, bias: bool = False):
        k, d = _pair(k), _pair(d)
        p = ((k[0] - 1) // 2 * d[0], (k[1] - 1) // 2 * d[1])
        layers = [nn.Conv2d(cin, cout, k, s, p, dilation=d, groups=g, bias=bias),
                  nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.PReLU(cout))
        super().__init__(*layers)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class CombCoef(nn.Module):
    """Combination Coefficient (CC).

    Weights the output of one branch before it is injected into another branch.
    The paper initialises it to 0.5 and learns it. Modes:
      * ``learn`` – learnable scalar, init 0.5 (paper setting)
      * ``fixed`` – constant 0.5 (ablation: interaction without learning)
      * ``off``   – 0, i.e. no cross-branch interaction (ablation)
    """

    def __init__(self, init: float = 0.5, mode: str = "learn"):
        super().__init__()
        assert mode in ("learn", "fixed", "off"), mode
        self.mode = mode
        if mode == "learn":
            self.alpha = nn.Parameter(torch.tensor(float(init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(init) if mode == "fixed" else 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return torch.zeros_like(x)
        return self.alpha.to(x.dtype) * x

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


class ECA(nn.Module):
    """Efficient Channel Attention (1-D conv over pooled channel descriptor)."""

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)                                   # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))     # B,1,C
        y = torch.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention (as used inside CAB)."""

    def __init__(self, c: int, squeeze: int = 16):
        super().__init__()
        mid = max(c // squeeze, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


def resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
