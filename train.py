"""Train LMIINet or a U-Net baseline on CamVid.

Recipe (LMIINet paper, CamVid): Adam (beta1 = 0.9), lr 1e-3, weight decay 2e-4, batch 8,
poly decay (power 0.9), cross-entropy + 0.3 x auxiliary cross-entropy at 1/8 resolution.

Speed: CamVid is held on the GPU and augmented there in batch (src/data/gpu.py), training runs in
bf16 autocast with cuDNN autotuning. --cuda-graph replays each step as a CUDA graph (tested equal
to eager training) but is off by default; see its help text.

    python train.py --model lmiinet --epochs 80
    python train.py --model unet --epochs 80
    python train.py --model lmiinet_no_cc --epochs 80             # ablation
    python train.py --model lmiinet --train-fraction 0.5           # reduced-data study
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import IGNORE_INDEX, NUM_CLASSES, CamVid, enet_class_weights, eval_transform, train_transform
from src.data.camvid import enet_weights_from_masks
from src.data.gpu import GPUEvalLoader, GPUTrainLoader, load_split
from src.engine import GraphTrainer, evaluate_model
from src.models import MODEL_ZOO, build_model
from src.utils import (count_macs, count_params, device_name, measure_fps, measure_fps_cudagraph,
                       pick_device, poly_lr, save_json, seed_worker, set_seed)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="lmiinet", choices=list(MODEL_ZOO))
    p.add_argument("--data-root", default="data/CamVid")
    p.add_argument("--out", default=None, help="run directory (default runs/<name>)")
    p.add_argument("--name", default=None, help="run name (default: model[_fracX][_seedY])")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--crop", type=int, nargs=2, default=[360, 480], metavar=("H", "W"))
    p.add_argument("--scale", type=float, nargs=2, default=[0.75, 1.75], metavar=("LO", "HI"))
    p.add_argument("--no-jitter", action="store_true")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="use only this fraction of the training images (reduced-data study)")
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--class-weights", default="enet", choices=["none", "enet"])
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="resize inputs at evaluation (e.g. 720 960); labels stay native")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--gpu-data", action=argparse.BooleanOptionalAction, default=True,
                   help="keep CamVid on the GPU and augment there in batch (default on)")
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False,
                   help="replay each training step as a CUDA graph. Off by default: capture is only valid "
                        "with cuDNN autotuning disabled, and on the RTX 3060 laptop the heuristic kernels "
                        "that leaves are far slower than the launch overhead the graph saves")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="resume from <out>/last.pt")
    p.add_argument("--fps-iters", type=int, default=200)
    return p.parse_args(argv)


def main(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (args.precision == "bf16" and cuda) else None

    name = args.name or args.model + (f"_frac{args.train_fraction:g}" if args.train_fraction < 1 else "") \
        + (f"_seed{args.seed}" if args.seed != 42 else "")
    out = args.out or os.path.join("runs", name)
    os.makedirs(out, exist_ok=True)
    print(f"==> run '{name}' on {device_name(device)} -> {out}", flush=True)

    # ---------------- data ----------------
    if cuda and args.gpu_data:
        # whole split resident on the GPU, augmentation done there in batch (src/data/gpu.py)
        tr_imgs, tr_masks, _ = load_split(args.data_root, ["train"], device, args.train_fraction, args.seed)
        va_imgs, va_masks, va_names = load_split(args.data_root, ["val"], device)
        train_loader = GPUTrainLoader(tr_imgs, tr_masks, args.batch_size, args.crop, args.scale,
                                      jitter=not args.no_jitter, seed=args.seed)
        val_loader = GPUEvalLoader(va_imgs, va_masks, va_names, 8, args.eval_size)
        n_train, n_val = tr_imgs.shape[0], va_imgs.shape[0]
        weight = enet_weights_from_masks(tr_masks) if args.class_weights == "enet" else None
    else:
        train_set = CamVid(args.data_root, "train", train_transform(args.crop, args.scale, not args.no_jitter),
                           fraction=args.train_fraction, seed=args.seed)
        val_set = CamVid(args.data_root, "val", eval_transform(args.eval_size))
        g = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.workers, pin_memory=cuda, worker_init_fn=seed_worker,
                                  generator=g, persistent_workers=args.workers > 0)
        val_loader = DataLoader(val_set, 8, shuffle=False, num_workers=args.workers, pin_memory=cuda,
                                persistent_workers=args.workers > 0)
        n_train, n_val = len(train_set), len(val_set)
        weight = enet_class_weights(train_set) if args.class_weights == "enet" else None
    print(f"    train {n_train} images (fraction {args.train_fraction}), val {n_val}, "
          f"data on {'GPU' if cuda and args.gpu_data else 'CPU workers'}", flush=True)

    # ---------------- model / loss / trainer ----------------
    model = build_model(args.model, NUM_CLASSES).to(device)
    if cuda:
        torch.backends.cudnn.benchmark = True      # fixed 360x480 crops: let cuDNN pick its kernels
        torch.cuda.reset_peak_memory_stats()
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss(weight=None if weight is None else weight.to(device),
                                    ignore_index=IGNORE_INDEX)
    trainer = GraphTrainer(model, criterion, model.parameters(), args.lr, args.weight_decay,
                           aux_weight=args.aux_weight, amp_dtype=amp_dtype,
                           use_graph=args.cuda_graph, device=device)
    print(f"    {args.model}: {n_params:,} params | precision {args.precision} | "
          f"step mode {trainer.mode}", flush=True)

    iters_per_epoch = len(train_loader)
    max_iter = args.epochs * iters_per_epoch
    start_epoch, best_miou, it = 0, -1.0, 0
    ckpt_last, ckpt_best = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    if args.resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        trainer.opt.load_state_dict(ck["optimizer"])
        start_epoch, best_miou = ck["epoch"] + 1, ck["best_miou"]
        it = start_epoch * iters_per_epoch
        print(f"    resumed at epoch {start_epoch + 1}, best val mIoU {best_miou:.2f}", flush=True)

    log_path = os.path.join(out, "log.csv")
    if not (args.resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_mIoU", "val_pixel_acc", "epoch_time_s"])

    def payload(epoch):
        return {"model": model.state_dict(), "optimizer": trainer.opt.state_dict(), "epoch": epoch,
                "best_miou": best_miou, "model_name": args.model, "num_classes": NUM_CLASSES,
                "args": vars(args)}

    # ---------------- train ----------------
    t_start = time.time()
    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        for batch in train_loader:
            imgs, masks = batch[0], batch[1]
            lr = poly_lr(args.lr, it, max_iter)
            trainer.set_lr(lr)
            trainer.step(imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True))
            it += 1
        train_loss = trainer.pop_mean_loss()
        dt = time.time() - t0
        row = [epoch + 1, f"{lr:.6f}", f"{train_loss:.4f}", "", "", f"{dt:.1f}"]
        msg = f"[{epoch + 1:4d}/{args.epochs}] loss {train_loss:.4f} lr {lr:.2e} | {dt:.1f}s"
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            res = evaluate_model(model, val_loader, device, use_amp=cuda)
            row[3], row[4] = f"{res['mIoU']:.2f}", f"{res['pixel_acc']:.2f}"
            if res["mIoU"] > best_miou:
                best_miou = res["mIoU"]
                torch.save(payload(epoch), ckpt_best)
            torch.save(payload(epoch), ckpt_last)
            msg += f" | val mIoU {res['mIoU']:.2f} (best {best_miou:.2f})"
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(msg, flush=True)
    train_minutes = (time.time() - t_start) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0

    # ---------------- final test evaluation of the best-on-val checkpoint ----------------
    ck = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if cuda and args.gpu_data:
        te_imgs, te_masks, te_names = load_split(args.data_root, ["test"], device)
        test_loader = GPUEvalLoader(te_imgs, te_masks, te_names, 8, args.eval_size)
    else:
        test_loader = DataLoader(CamVid(args.data_root, "test", eval_transform(args.eval_size)), 8,
                                 num_workers=args.workers)
    test = evaluate_model(model, test_loader, device, use_amp=cuda)
    print(f"==> TEST (best val checkpoint, epoch {ck['epoch'] + 1}): mIoU {test['mIoU']:.2f} | "
          f"pixel acc {test['pixel_acc']:.2f}", flush=True)

    # ---------------- efficiency ----------------
    size = tuple(args.eval_size) if args.eval_size else (360, 480)
    macs = count_macs(model, size)
    fps = measure_fps(model, size, device, iters=args.fps_iters) if cuda else None
    fps_graph = measure_fps_cudagraph(model, size, iters=args.fps_iters) if cuda else None
    learned_cc = {n: round(float(p.detach()), 4) for n, p in model.named_parameters() if n.endswith(".alpha")}
    results = {
        "run": name, "model": args.model, "params": n_params, "params_M": n_params / 1e6,
        "gmacs": macs / 1e9, "fps": fps, "fps_graph": fps_graph, "fps_input": list(size),
        "device": device_name(device), "test": test, "best_monitor_mIoU": best_miou,
        "monitor_split": "val", "checkpoint": "best.pt", "epoch_selected": ck["epoch"] + 1,
        "learned_cc": learned_cc, "peak_train_memory_gb": peak_gb, "train_minutes": train_minutes,
        "step_mode": trainer.mode, "args": vars(args),
    }
    save_json(results, os.path.join(out, "results.json"))
    print(f"    {n_params / 1e6:.3f}M params | {macs / 1e9:.2f} GMACs | FPS eager {fps or 0:.1f}, "
          f"CUDA graph {fps_graph or 0:.1f} | trained in {train_minutes:.1f} min | peak {peak_gb:.2f} GB",
          flush=True)
    return results


if __name__ == "__main__":
    main(get_args())
