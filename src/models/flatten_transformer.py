"""Improved Flatten Transformer block of LMIINet (Eqs. 7-12).

Focused Linear Attention follows FLatten Transformer (Han et al., ICCV 2023):
    phi_p(x) = ||x|| / ||x^p|| * x^p   with x = ReLU(x)          (8)
    y        = phi_p(Q) (phi_p(K)^T V) * Z + DWC(V)              (7)
which costs O(N d^2) instead of O(N^2 d), so it is cheap at 1/8 resolution.

LMIINet's modification wraps the attention with a Channel Attention Block (CAB)
and a combination coefficient (CC):
    F11 = FLAM(LN(x))                                            (9)
    F12 = CAB(x + CC(F11))                                       (10)
    F1  = F11 + F12 + x                                          (11)
    y   = F1 + MLP(LN(F1))                                       (12)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ChannelAttention, CombCoef


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Row-normalised kernel attention in O(N d^2).

    With non-negative feature maps q, k (B,h,N,d) this equals
        diag(1 / (q k^T 1)) (q k^T) v
    i.e. attention weights q_i.k_j / sum_j q_i.k_j, without forming the N x N matrix.
    The 1/sqrt(N) factors keep k^T v in a safe range; they cancel against Z. The 1e-6 in Z is
    FLatten's stabiliser: after focusing over all channels a head can receive an almost-zero
    query, and for such a head the stabiliser (not the data) sets the scale of its output.
    """
    n = q.shape[-2]
    z = 1.0 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)   # B,h,N,1
    kv = (k.transpose(-2, -1) * n ** -0.5) @ (v * n ** -0.5)                  # B,h,d,d
    return (q @ kv) * z


def focus(x: torch.Tensor, p: float) -> torch.Tensor:
    """Focusing function phi_p (Eq. 8): raise to the power p, then restore the original norm."""
    x_norm = x.norm(dim=-1, keepdim=True)
    x = x ** p
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6) * x_norm


def _f32(t: torch.Tensor) -> torch.Tensor:
    """Promote half precision to fp32; leave fp32/fp64 untouched (tests run in fp64)."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, focusing_factor: float = 3.0,
                 kernel_size: int = 5, qkv_bias: bool = True, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dwc = nn.Conv2d(self.head_dim, self.head_dim, kernel_size,
                             padding=kernel_size // 2, groups=self.head_dim)
        self.scale = nn.Parameter(torch.zeros(1, 1, dim))

    def matmul_macs(self, n_tokens: int) -> int:
        """MACs of the attention matmuls (not visible to Conv/Linear hooks)."""
        d = self.head_dim
        return self.num_heads * (2 * n_tokens * d * d + 2 * n_tokens * d)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        dev = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        # The power / normalisation steps are numerically fragile in fp16 -> fp32.
        with torch.autocast(device_type=dev, enabled=False):
            q, k, v = _f32(q), _f32(k), _f32(v)
            scale = F.softplus(_f32(self.scale)).to(q.dtype)
            q = (F.relu(q) + 1e-6) / scale
            k = (F.relu(k) + 1e-6) / scale
            q, k = focus(q, self.focusing_factor), focus(k, self.focusing_factor)

            h, d = self.num_heads, self.head_dim
            q = q.reshape(B, N, h, d).transpose(1, 2)          # B,h,N,d
            k = k.reshape(B, N, h, d).transpose(1, 2)
            v = v.reshape(B, N, h, d).transpose(1, 2)

            out = linear_attention(q, k, v)                                         # B,h,N,d

            fmap = v.reshape(B * h, H, W, d).permute(0, 3, 1, 2)                  # B*h,d,H,W
            fmap = self.dwc(fmap.to(self.dwc.weight.dtype)).to(out.dtype).permute(0, 2, 3, 1).reshape(B, h, N, d)
            out = (out + fmap).transpose(1, 2).reshape(B, N, C)

        return self.proj_drop(self.proj(out.to(self.proj.weight.dtype)))


class CAB(nn.Module):
    """Channel Attention Block: conv3x3 -> GELU -> conv3x3 -> channel attention."""

    def __init__(self, c: int, compress: int = 4, squeeze: int = 16):
        super().__init__()
        mid = max(c // compress, 8)
        self.body = nn.Sequential(
            nn.Conv2d(c, mid, 3, 1, 1), nn.GELU(),
            nn.Conv2d(mid, c, 3, 1, 1), ChannelAttention(c, squeeze))

    def forward(self, x):
        return self.body(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class FlattenTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 focusing_factor: float = 3.0, kernel_size: int = 5,
                 cab_compress: int = 4, cab_squeeze: int = 16, cc_mode: str = "learn",
                 use_cab: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FocusedLinearAttention(dim, num_heads, focusing_factor, kernel_size)
        self.cc = CombCoef(0.5, cc_mode) if use_cab else None
        self.cab = CAB(dim, cab_compress, cab_squeeze) if use_cab else None
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                                   # B,N,C
        f11 = self.attn(self.norm1(t), H, W).transpose(1, 2).reshape(B, C, H, W)
        f1 = f11 + x
        if self.cab is not None:
            f1 = f1 + self.cab(x + self.cc(f11))                           # (10)-(11)
        t1 = f1.flatten(2).transpose(1, 2)
        t1 = t1 + self.mlp(self.norm2(t1))                                 # (12)
        return t1.transpose(1, 2).reshape(B, C, H, W)
