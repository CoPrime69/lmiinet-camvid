"""LMIINet: Lightweight Multiple-Information Interaction Network.

Topology (UNet-like encoder-decoder, 6 LFIB stages x 2 blocks = {2,2,2,2,2,2}):

    input H x W
      Stem (3x conv3x3, s2)                     32 ch  @1/2
      E1: LFIB x2 (d=1,2)        ----LC1----+   32 ch  @1/2
      Down 32->64                           |
      E2: LFIB x2 (d=2,4)        ----LC2-+  |   64 ch  @1/4
      Down 64->128                       |  |
      E3: LFIB x2 (d=4,8)                |  |  128 ch  @1/8
      Improved Flatten Transformer x1    |  |  128 ch  @1/8
      D3: LFIB x2 (d=4,2)  --> aux head  |  |  128 ch  @1/8
      FE(D3, LC2)  -> D2: LFIB x2 (d=2,1)   |   64 ch  @1/4
      FE(D2, LC1)  -> D1: LFIB x2 (d=1,1)       32 ch  @1/2
      SegHead(concat[D1, up(D2), up(D3)]) -> upsample x2 -> H x W

~0.72M parameters with the default configuration (the paper reports 0.72M).
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flatten_transformer import FlattenTransformerBlock
from .modules import (DownsamplingBlock, FeatureEnhancement, LongConnection,
                      SegHead, Stem, lfib_stage)


class LMIINet(nn.Module):
    def __init__(self, num_classes: int = 11,
                 channels: Sequence[int] = (32, 64, 128),
                 enc_dilations: Sequence[Sequence[int]] = ((1, 2), (2, 4), (4, 8)),
                 dec_dilations: Sequence[Sequence[int]] = ((4, 2), (2, 1), (1, 1)),
                 tf_blocks: int = 1, tf_heads: int = 4, tf_mlp_ratio: float = 2.0,
                 focusing_factor: float = 3.0, cc_mode: str = "learn",
                 use_cru: bool = True, use_cab: bool = True, use_fe: bool = True,
                 aux_head: bool = True, head_mid: int = 64):
        super().__init__()
        c1, c2, c3 = channels
        kw = dict(cc_mode=cc_mode, use_cru=use_cru)
        self.use_fe = use_fe

        # ---------------- encoder ----------------
        self.stem = Stem(3, c1)
        self.enc1 = lfib_stage(c1, enc_dilations[0], **kw)
        self.down1 = DownsamplingBlock(c1, c2)
        self.enc2 = lfib_stage(c2, enc_dilations[1], **kw)
        self.down2 = DownsamplingBlock(c2, c3)
        self.enc3 = lfib_stage(c3, enc_dilations[2], **kw)

        # ---------------- global context ----------------
        self.transformer = nn.Sequential(*[
            FlattenTransformerBlock(c3, tf_heads, tf_mlp_ratio, focusing_factor,
                                    cc_mode=cc_mode, use_cab=use_cab)
            for _ in range(tf_blocks)]) if tf_blocks > 0 else nn.Identity()

        # ---------------- decoder ----------------
        self.dec3 = lfib_stage(c3, dec_dilations[0], **kw)
        self.lc2 = LongConnection(c2, c2)
        self.lc1 = LongConnection(c1, c1)
        if use_fe:
            self.fe2 = FeatureEnhancement(c3, c2, c2)
            self.fe1 = FeatureEnhancement(c2, c1, c1)
        else:  # ablation: plain "reduce + upsample + add" skip fusion
            self.fe2 = nn.Conv2d(c3, c2, 1, bias=False)
            self.fe1 = nn.Conv2d(c2, c1, 1, bias=False)
        self.dec2 = lfib_stage(c2, dec_dilations[1], **kw)
        self.dec1 = lfib_stage(c1, dec_dilations[2], **kw)

        self.head = SegHead([c1, c2, c3], num_classes, mid=head_mid)
        self.aux = nn.Conv2d(c3, num_classes, 1) if aux_head else None

        self._init_weights()

    def set_grad_checkpointing(self, on: bool = True):
        """Recompute LFIB stages and the Transformer during backward instead of storing them."""
        for m in self.modules():
            if hasattr(m, "checkpoint") and isinstance(m, nn.Sequential):
                m.checkpoint = on
        self._ckpt_transformer = on
        return self

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fuse(self, fe, high, low):
        if self.use_fe:
            return fe(high, low)
        h = fe(high)
        return F.interpolate(h, size=low.shape[-2:], mode="bilinear", align_corners=False) + low

    def forward(self, x: torch.Tensor):
        size = x.shape[-2:]
        s = self.stem(x)
        e1 = self.enc1(s)                      # 1/2
        e2 = self.enc2(self.down1(e1))         # 1/4
        e3 = self.enc3(self.down2(e2))         # 1/8

        if getattr(self, "_ckpt_transformer", False) and self.training and torch.is_grad_enabled() \
                and not isinstance(self.transformer, nn.Identity):
            from torch.utils.checkpoint import checkpoint
            t = checkpoint(self.transformer, e3, use_reentrant=False)
        else:
            t = self.transformer(e3)
        d3 = self.dec3(t)                      # 1/8
        d2 = self.dec2(self._fuse(self.fe2, d3, self.lc2(e2)))   # 1/4
        d1 = self.dec1(self._fuse(self.fe1, d2, self.lc1(e1)))   # 1/2

        out = self.head([d1, d2, d3])
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        if self.training and self.aux is not None:
            aux = F.interpolate(self.aux(d3), size=size, mode="bilinear", align_corners=False)
            return out, aux
        return out
