"""Qualitative figure: image | ground truth | prediction of each model, for a few test images.

Writes figures/<name>.png/.pdf and results/<name>.csv. The CSV holds, for every panel shown,
the per-image mIoU and pixel accuracy of that prediction (the numbers that describe the figure).

    python -m src.visualize --ckpts runs/lmiinet/best.pt runs/unet/best.pt --names LMIINet U-Net
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from .data import CAMVID_CLASSES, CAMVID_PALETTE, NUM_CLASSES, CamVid, colorize, eval_transform
from .data.camvid import MEAN, STD
from .engine import load_checkpoint
from .utils import ConfusionMatrix, pick_device


def qualitative_figure(ckpts, names, data_root="data/CamVid", indices=(10, 70, 130, 200),
                       device="auto", fig_dir="figures", res_dir="results", stem="04_qualitative"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    device = pick_device(device) if isinstance(device, str) else device
    models = [load_checkpoint(c, device)[0] for c in ckpts]
    ds = CamVid(data_root, "test", eval_transform())
    indices = [i for i in indices if i < len(ds)]

    cols = 2 + len(models)
    fig, axes = plt.subplots(len(indices), cols, figsize=(1.9 * cols, 1.45 * len(indices)), squeeze=False)
    mean, std = np.array(MEAN), np.array(STD)
    rows = []
    for r, i in enumerate(indices):
        img, mask, fname = ds[i]
        rgb = np.clip(img.permute(1, 2, 0).numpy() * std + mean, 0, 1)
        panels = [rgb, colorize(mask.numpy())]
        with torch.no_grad():
            for name, m in zip(names, models):
                logit = m(img.unsqueeze(0).to(device))
                logit = F.interpolate(logit, size=tuple(mask.shape), mode="bilinear", align_corners=False)
                pred = logit.argmax(1)[0]
                cm = ConfusionMatrix(NUM_CLASSES, device)
                cm.update(pred, mask.to(device))
                res = cm.compute()
                rows.append([fname, name, f"{res['mIoU']:.2f}", f"{res['pixel_acc']:.2f}"])
                panels.append(colorize(pred.cpu().numpy()))
        for c, pnl in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(pnl)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title((["Image", "Ground truth"] + list(names))[c], fontsize=8)
    handles = [Patch(color=CAMVID_PALETTE[k] / 255, label=n) for k, n in enumerate(CAMVID_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, stem + ".pdf"))
    fig.savefig(os.path.join(fig_dir, stem + ".png"), dpi=200)
    plt.close(fig)
    with open(os.path.join(res_dir, stem + ".csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["test_image", "model", "image_mIoU", "image_pixel_acc"])
        wr.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--data-root", default="data/CamVid")
    a = p.parse_args()
    qualitative_figure(a.ckpts, a.names, a.data_root)
