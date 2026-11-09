"""Efficiency benchmark: parameters, MACs and FPS for every model. Needs no data or weights.

    python -m src.benchmark                     # writes results/benchmark.json and results/benchmark.csv
"""
import argparse
import csv
import os

import torch

from .models import build_model
from .utils import (count_macs, count_params, device_name, measure_fps, measure_fps_cudagraph,
                     pick_device, save_json)

MODELS = ["lmiinet", "unet", "unet_pm", "lmiinet_no_cc", "lmiinet_fixed_cc", "lmiinet_no_cru",
          "lmiinet_no_tf", "lmiinet_no_cab", "lmiinet_no_fe", "lmiinet_no_aux"]
SIZES = [(360, 480), (720, 960)]


MAIN_MODELS = ["lmiinet", "unet", "unet_pm"]


def run_benchmark(models=MODELS, sizes=SIZES, device="auto", iters=200, warmup=50,
                  half=True, out_dir="results", log=print, hires_models=MAIN_MODELS):
    device = pick_device(device) if isinstance(device, str) else device
    dev_name = device_name(device)
    if device.type != "cuda":
        iters, warmup = max(5, iters // 20), 3
    log(f"benchmark on {dev_name} ({iters} timed passes after {warmup} warm-up)")
    rows = []
    for name in models:
        model = build_model(name).to(device).eval()
        n = count_params(model)
        for (h, w) in sizes:
            if (h, w) != sizes[0] and name not in hires_models:
                continue
            macs = count_macs(model, (h, w))
            fps = measure_fps(model, (h, w), device, warmup=warmup, iters=iters)
            cuda = device.type == "cuda"
            fps16 = measure_fps(model, (h, w), device, warmup=warmup, iters=iters, half=True) \
                if half and cuda else None
            fps_g = measure_fps_cudagraph(model, (h, w), iters=iters) if cuda else None
            fps_g16 = measure_fps_cudagraph(model, (h, w), iters=iters, half=True) if half and cuda else None
            rows.append({"model": name, "params": n, "params_M": n / 1e6, "input": [h, w],
                         "gmacs": macs / 1e9, "fps": fps, "fps_fp16": fps16, "fps_graph": fps_g,
                         "fps_graph_fp16": fps_g16, "device": dev_name})
            log(f"  {name:<17s} {h}x{w}  {n / 1e6:6.3f}M  {macs / 1e9:6.2f} GMACs | eager {fps:6.1f} "
                f"(fp16 {fps16 or 0:6.1f}) | CUDA graph {fps_g or 0:7.1f} (fp16 {fps_g16 or 0:7.1f}) FPS")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(out_dir, exist_ok=True)
    save_json({"device": dev_name, "rows": rows}, os.path.join(out_dir, "benchmark.json"))
    with open(os.path.join(out_dir, "benchmark.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "input_h", "input_w", "params", "gmacs", "fps_eager_fp32", "fps_eager_fp16",
                     "fps_cudagraph_fp32", "fps_cudagraph_fp16", "device"])
        f2 = lambda v: "" if v is None else f"{v:.2f}"
        for r in rows:
            wr.writerow([r["model"], *r["input"], r["params"], f"{r['gmacs']:.4f}", f2(r["fps"]),
                         f2(r["fps_fp16"]), f2(r["fps_graph"]), f2(r["fps_graph_fp16"]), r["device"]])
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--no-half", action="store_true")
    a = p.parse_args()
    run_benchmark(device=a.device, iters=a.iters, half=not a.no_half)
