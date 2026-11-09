"""Evaluate a checkpoint on a CamVid split (mIoU, per-class IoU, pixel accuracy).

    python evaluate.py --ckpt runs/lmiinet/best.pt --split test
    python evaluate.py --ckpt runs/lmiinet/best.pt --split test --eval-size 720 960
"""
import argparse
import os

from torch.utils.data import DataLoader

from src.data import CAMVID_CLASSES, CamVid, eval_transform
from src.engine import evaluate_model, load_checkpoint
from src.utils import pick_device, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="data/CamVid")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--eval-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="json path (default next to ckpt)")
    a = p.parse_args()

    device = pick_device(a.device)
    model, ck = load_checkpoint(a.ckpt, device)
    ds = CamVid(a.data_root, a.split, eval_transform(a.eval_size))
    res = evaluate_model(model, DataLoader(ds, 1, num_workers=2), device)
    print(f"{ck['model_name']} | {a.split} | mIoU {res['mIoU']:.2f} | pixel acc {res['pixel_acc']:.2f}")
    for c, v in zip(CAMVID_CLASSES, res["per_class_iou"]):
        print(f"  {c:<11s} {'n/a' if v is None else f'{v:6.2f}'}")
    tag = f"{a.split}" + (f"_{a.eval_size[0]}x{a.eval_size[1]}" if a.eval_size else "")
    out = a.out or os.path.join(os.path.dirname(a.ckpt), f"eval_{tag}.json")
    save_json({"model": ck["model_name"], "split": a.split, "eval_size": a.eval_size, **res}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
