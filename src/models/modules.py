"""CNN modules of LMIINet: CRU, LFIB, down-sampling, long connections, FE, SegHead.

Equation numbers refer to Qiu et al., "Efficient Semantic Segmentation via
Lightweight Multiple-Information Interaction Network" (arXiv:2410.02224).
Where the paper leaves a detail unspecified, the choice made here is stated
in the docstring (see also docs/DESIGN_DECISIONS.md).
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ECA, CombCoef, ConvBNAct, channel_shuffle, resize_to


class CRU(nn.Module):
    """Channel Reconstruction Unit from SCConv (Li et al., CVPR 2023).

    Split -> (squeeze) -> transform (GWC + PWC for the 'rich' half, cheap PWC +
    identity reuse for the other half) -> softmax-weighted fuse.
    """

    def __init__(self, c: int, alpha: float = 0.5, squeeze_ratio: int = 2,
                 groups: int = 2, k: int = 3):
        super().__init__()
        self.up_c = int(alpha * c)
        self.low_c = c - self.up_c
        up_s, low_s = self.up_c // squeeze_ratio, self.low_c // squeeze_ratio
        assert up_s % groups == 0 and c % groups == 0, "CRU: channels must divide groups"
        self.squeeze_up = nn.Conv2d(self.up_c, up_s, 1, bias=False)
        self.squeeze_low = nn.Conv2d(self.low_c, low_s, 1, bias=False)
        self.gwc = nn.Conv2d(up_s, c, k, padding=k // 2, groups=groups)
        self.pwc_up = nn.Conv2d(up_s, c, 1, bias=False)
        self.pwc_low = nn.Conv2d(low_s, c - low_s, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, low = torch.split(x, [self.up_c, self.low_c], dim=1)
        up, low = self.squeeze_up(up), self.squeeze_low(low)
        y1 = self.gwc(up) + self.pwc_up(up)
        y2 = torch.cat([self.pwc_low(low), low], dim=1)
        out = torch.cat([y1, y2], dim=1)
        out = F.softmax(self.pool(out), dim=1) * out
        o1, o2 = torch.split(out, out.size(1) // 2, dim=1)
        return o1 + o2


class LFIB(nn.Module):
    """Lightweight Feature Interaction Bottleneck (Eqs. 1-6).

        F1        = H_1x1(x)                                   (1)
        F21, F22  = Split(F1)                                  (2)
        F21'      = H_1x3(H_3x1(F21 + CC(F22)))                (3)  local / asymmetric
        F22'      = H_1x1(H_3x3DW,d(F22 + CC(F21')))           (4)  context / dilated DW + PW
        F2        = Concat(F21', F22') + CRU(F1)               (5)
        y         = ChannelShuffle(H_3x3DW(x + F2))            (6)

    Choices for unspecified details: the right branch receives the *updated*
    left output F21' (sequential interaction), the CC is a learnable scalar
    per injection, channel shuffle uses 2 groups.
    """

    def __init__(self, c: int, dilation: int = 1, cc_mode: str = "learn",
                 use_cru: bool = True, shuffle_groups: int = 2):
        super().__init__()
        assert c % 2 == 0
        h = c // 2
        self.conv_in = ConvBNAct(c, c, 1)
        self.cc_left = CombCoef(0.5, cc_mode)
        self.cc_right = CombCoef(0.5, cc_mode)
        self.left = nn.Sequential(ConvBNAct(h, h, (3, 1)), ConvBNAct(h, h, (1, 3)))
        self.right = nn.Sequential(ConvBNAct(h, h, 3, d=dilation, g=h), ConvBNAct(h, h, 1))
        self.cru = CRU(c) if use_cru else None
        self.conv_out = ConvBNAct(c, c, 3, g=c)
        self.shuffle_groups = shuffle_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv_in(x)
        a, b = torch.chunk(f1, 2, dim=1)
        left = self.left(a + self.cc_left(b))
        right = self.right(b + self.cc_right(left))
        f2 = torch.cat([left, right], dim=1)
        if self.cru is not None:
            f2 = f2 + self.cru(f1)
        y = self.conv_out(x + f2)
        return channel_shuffle(y, self.shuffle_groups)


class CheckpointedSequential(nn.Sequential):
    """nn.Sequential that, when ``checkpoint`` is on and the model is training, stores only each
    block's input and recomputes the block during backward. Results are identical; peak memory
    drops sharply (an LFIB keeps ~30 intermediate maps), at the price of one extra forward."""

    checkpoint = False

    def forward(self, x):
        if not (self.checkpoint and self.training and torch.is_grad_enabled()):
            return super().forward(x)
        from torch.utils.checkpoint import checkpoint
        for block in self:
            x = checkpoint(block, x, use_reentrant=False)
        return x


def lfib_stage(c: int, dilations: List[int], **kw) -> CheckpointedSequential:
    return CheckpointedSequential(*[LFIB(c, d, **kw) for d in dilations])


class Stem(nn.Module):
    """Initial block: three 3x3 convs, the first with stride 2 (as in LETNet)."""

    def __init__(self, cin: int = 3, c: int = 32):
        super().__init__()
        self.body = nn.Sequential(ConvBNAct(cin, c, 3, s=2), ConvBNAct(c, c, 3), ConvBNAct(c, c, 3))

    def forward(self, x):
        return self.body(x)


class DownsamplingBlock(nn.Module):
    """ENet/LETNet style: concat(3x3 stride-2 conv, 2x2 max-pool) -> BN -> PReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.use_pool = cin < cout
        self.conv = nn.Conv2d(cin, cout - cin if self.use_pool else cout, 3, 2, 1, bias=False)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x):
        y = self.conv(x)
        if self.use_pool:
            p = self.pool(x)
            if p.shape[-2:] != y.shape[-2:]:
                p = F.interpolate(p, size=y.shape[-2:], mode="nearest")
            y = torch.cat([y, p], dim=1)
        return self.act(self.bn(y))


class LongConnection(nn.Module):
    """Long connection: carries encoder detail (boundary) features to the decoder.

    Implemented as a light depthwise-separable refinement of the encoder feature.
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(ConvBNAct(cin, cin, 3, g=cin), ConvBNAct(cin, cout, 1))

    def forward(self, x):
        return self.body(x)


class FeatureEnhancement(nn.Module):
    """FE (after LETNet): fuse up-sampled semantic features with encoder details.

    high -> 1x1 reduce -> upsample; a spatial attention map computed from the
    semantic features gates the detail features (where to add boundary detail),
    then the sum is refined by a depthwise conv and ECA channel attention.
    """

    def __init__(self, high_c: int, low_c: int, out_c: int):
        super().__init__()
        self.reduce = ConvBNAct(high_c, out_c, 1)
        self.align = ConvBNAct(low_c, out_c, 1) if low_c != out_c else nn.Identity()
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.refine = ConvBNAct(out_c, out_c, 3, g=out_c)
        self.eca = ECA(out_c)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        h = resize_to(self.reduce(high), low)
        l = self.align(low)
        s = self.spatial(torch.cat([h.mean(1, keepdim=True), h.amax(1, keepdim=True)], dim=1))
        return self.eca(self.refine(h + l * s))


class SegHead(nn.Module):
    """Multi-scale segmentation head (Eqs. 13-14).

    Features from 1/2, 1/4 and 1/8 are up-sampled to the 1/2 resolution and
    concatenated (the text says concat; Eq. 13 writes a sum), then two 1x1 convs.
    """

    def __init__(self, in_channels: List[int], num_classes: int, mid: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fuse = ConvBNAct(sum(in_channels), mid, 1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls = nn.Conv2d(mid, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        ref = feats[0]
        x = torch.cat([resize_to(f, ref) for f in feats], dim=1)
        return self.cls(self.drop(self.fuse(x)))
