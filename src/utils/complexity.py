"""Parameter count, MACs ("FLOPs" as reported by most segmentation papers) and FPS."""
import time
from typing import Sequence

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, input_size: Sequence[int] = (360, 480)) -> int:
    """Multiply-accumulates for one image, via forward hooks.

    Counts Conv2d / ConvTranspose2d / Linear plus the attention matmuls of
    FocusedLinearAttention (which expose ``matmul_macs``). Norms, activations
    and element-wise ops are ignored, matching thop/fvcore conventions.
    """
    total = [0]
    hooks = []

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        cin = m.in_channels // m.groups
        if isinstance(m, nn.ConvTranspose2d):
            total[0] += inp[0].numel() * (m.out_channels // m.groups) * k
        else:
            total[0] += out.numel() * cin * k

    def linear_hook(m, inp, out):
        total[0] += out.numel() * m.in_features

    def attn_hook(m, inp, out):
        total[0] += inp[0].shape[0] * m.matmul_macs(inp[0].shape[1])

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif hasattr(m, "matmul_macs"):
            hooks.append(m.register_forward_hook(attn_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(1, 3, *input_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total[0]


@torch.no_grad()
def measure_fps_cudagraph(model: nn.Module, input_size=(360, 480), batch: int = 1,
                          warmup: int = 20, iters: int = 200, half: bool = False) -> float:
    """FPS with the forward pass captured once as a CUDA graph and replayed.

    Replaying a graph removes the per-kernel CPU launch cost, which dominates eager PyTorch for
    networks made of many small kernels. The arithmetic executed on the GPU is identical.
    """
    device = torch.device("cuda")
    model = model.to(device).eval()
    x = torch.randn(batch, 3, *input_size, device=device)
    if half:
        model, x = model.half(), x.half()
    torch.backends.cudnn.benchmark = True
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            model(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model(x)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if half:
        model.float()
    return batch * iters / dt


@torch.no_grad()
def measure_fps(model: nn.Module, input_size=(360, 480), device="cuda", batch: int = 1,
                warmup: int = 50, iters: int = 200, half: bool = False) -> float:
    device = torch.device(device)
    model = model.to(device).eval()
    x = torch.randn(batch, 3, *input_size, device=device)
    if half and device.type == "cuda":
        model, x = model.half(), x.half()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if half and device.type == "cuda":
        model.float()
    return batch * iters / dt


def device_name(device) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    import platform
    return f"CPU ({platform.processor() or platform.machine()})"
