"""Correctness checks. Each one compares the implementation with an independent reference:
an explicit O(N^2) computation, a brute-force loop, a closed-form count, or a known answer.

    python -m src.tests            # prints a table and writes results/00_unit_tests.csv
run.py calls run_tests() before anything is trained and stops if a check fails.
"""
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IGNORE_INDEX, NUM_CLASSES
from .data.transforms import RandomCrop, RandomHFlip, RandomScale
from .models import MODEL_ZOO, build_model, has_aux
from .models.common import CombCoef, channel_shuffle
from .models.flatten_transformer import FocusedLinearAttention, focus, linear_attention
from .models.modules import CRU, LFIB, DownsamplingBlock
from .utils import ConfusionMatrix, count_macs, count_params, poly_lr

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ------------------------------------------------------------------ attention
@check
def linear_attention_equals_explicit_attention():
    """O(N d^2) form == row-normalised (q k^T) v built as an explicit N x N matrix."""
    g = torch.Generator().manual_seed(0)
    q = torch.rand(2, 4, 50, 8, generator=g, dtype=torch.float64)
    k = torch.rand(2, 4, 50, 8, generator=g, dtype=torch.float64)
    v = torch.randn(2, 4, 50, 8, generator=g, dtype=torch.float64)
    a = q @ k.transpose(-2, -1)
    ref = (a / a.sum(-1, keepdim=True)) @ v
    err = (linear_attention(q, k, v) - ref).abs().max().item()
    assert err < 1e-6, err
    return f"max abs error {err:.1e}"


@check
def focus_keeps_norm_and_sharpens():
    """phi_p keeps ||x|| and makes the vector more peaked (larger max/mean ratio)."""
    x = torch.rand(1000, 32, dtype=torch.float64) + 1e-3
    y = focus(x, 3.0)
    norm_err = (y.norm(dim=-1) - x.norm(dim=-1)).abs().max().item()
    peak_x = (x.max(-1).values / x.mean(-1)).mean().item()
    peak_y = (y.max(-1).values / y.mean(-1)).mean().item()
    assert norm_err < 1e-5 and peak_y > peak_x, (norm_err, peak_x, peak_y)
    return f"norm error {norm_err:.1e}, peak ratio {peak_x:.2f} -> {peak_y:.2f}"


@check
def focused_linear_attention_module_matches_reference():
    """Whole FLAM forward == a slow reference written from Eqs. 7-8 with the same weights."""
    torch.manual_seed(1)
    m = FocusedLinearAttention(32, num_heads=4).double().eval()
    nn.init.normal_(m.scale, std=0.3)
    B, H, W = 2, 5, 7
    x = torch.randn(B, H * W, 32, dtype=torch.float64)
    out = m(x, H, W)

    q, k, v = m.qkv(x).chunk(3, dim=-1)
    s = F.softplus(m.scale)
    q, k = focus((F.relu(q) + 1e-6) / s, 3.0), focus((F.relu(k) + 1e-6) / s, 3.0)
    ref = torch.zeros(B, H * W, 32, dtype=torch.float64)
    d = 8
    for b in range(B):
        for h in range(4):
            sl = slice(h * d, (h + 1) * d)
            qa, ka, va = q[b, :, sl], k[b, :, sl], v[b, :, sl]
            att = qa @ ka.T
            # same 1e-6 stabiliser as FLatten's Z, expressed on the un-averaged sum (N * 1e-6)
            y = (att @ va) / (att.sum(1, keepdim=True) + H * W * 1e-6)
            vmap = va.T.reshape(1, d, H, W)
            y = y + m.dwc(vmap)[0].reshape(d, H * W).T
            ref[b, :, sl] = y
    ref = m.proj(ref)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, err
    return f"max abs error {err:.1e}"


# ------------------------------------------------------------------ metrics
@check
def confusion_matrix_miou_equals_bruteforce():
    """mIoU from the bincount confusion matrix == per-class set counting, void ignored."""
    rng = np.random.default_rng(0)
    gt = rng.integers(0, NUM_CLASSES + 1, size=(4, 37, 53))       # includes void (=11)
    pred = np.where(rng.random(gt.shape) < 0.6, gt, rng.integers(0, NUM_CLASSES, size=gt.shape))
    pred = np.where(pred == IGNORE_INDEX, 0, pred)
    cm = ConfusionMatrix(NUM_CLASSES)
    cm.update(torch.from_numpy(pred), torch.from_numpy(gt))
    res = cm.compute()
    valid = gt != IGNORE_INDEX
    ious = []
    for c in range(NUM_CLASSES):
        inter = np.sum((pred == c) & (gt == c) & valid)
        union = np.sum(((pred == c) | (gt == c)) & valid)
        if union:
            ious.append(inter / union)
    ref = 100 * float(np.mean(ious))
    acc = 100 * float(np.sum((pred == gt) & valid) / valid.sum())
    assert abs(res["mIoU"] - ref) < 1e-9 and abs(res["pixel_acc"] - acc) < 1e-9
    return f"mIoU {res['mIoU']:.4f} == {ref:.4f}"


@check
def mac_counter_matches_closed_form():
    """Hook MAC count == hand formula for a conv, a depthwise conv and a linear layer."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8)
            self.fc = nn.Linear(8, 5)

        def forward(self, x):
            x = self.c2(self.c1(x))
            return self.fc(x.mean((2, 3)))
    macs = count_macs(Tiny(), (20, 30))
    ref = 20 * 30 * 8 * 3 * 9 + 10 * 15 * 8 * 1 * 9 + 8 * 5
    assert macs == ref, (macs, ref)
    return f"{macs} == {ref}"


@check
def poly_schedule_endpoints():
    lrs = [poly_lr(1e-3, i, 100) for i in (0, 50, 100)]
    assert abs(lrs[0] - 1e-3) < 1e-15 and abs(lrs[1] - 1e-3 * 0.5 ** 0.9) < 1e-15 and lrs[2] == 0
    return "1e-3 at start, 1e-3*0.5^0.9 halfway, 0 at the end"


# ------------------------------------------------------------------ building blocks
@check
def channel_shuffle_is_the_expected_permutation():
    x = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)
    y = channel_shuffle(x, 2).flatten().tolist()
    assert y == [0, 3, 1, 4, 2, 5], y
    return "[0..5] -> [0,3,1,4,2,5]"


@check
def combination_coefficient_modes():
    x = torch.ones(3)
    learn, fixed, off = CombCoef(0.5, "learn"), CombCoef(0.5, "fixed"), CombCoef(0.5, "off")
    assert isinstance(learn.alpha, nn.Parameter) and not isinstance(fixed.alpha, nn.Parameter)
    assert learn(x).tolist() == [0.5] * 3 and fixed(x).tolist() == [0.5] * 3 and off(x).tolist() == [0] * 3
    (learn(x).sum()).backward()
    assert learn.alpha.grad.item() == 3.0
    return "learn: trainable 0.5, fixed: buffer 0.5, off: 0"


@check
def cru_and_lfib_preserve_shape_and_backprop():
    for c in (32, 64, 128):
        x = torch.randn(2, c, 11, 13, requires_grad=True)
        y1, y2 = CRU(c)(x), LFIB(c, dilation=4)(x)
        assert y1.shape == x.shape and y2.shape == x.shape
        (y1.sum() + y2.sum()).backward()
        assert torch.isfinite(x.grad).all()
    return "C in {32, 64, 128}, odd 11x13 maps"


@check
def downsampling_handles_odd_sizes():
    for (h, w) in [(45, 60), (23, 30), (45, 59)]:
        y = DownsamplingBlock(32, 64)(torch.randn(1, 32, h, w))
        assert y.shape == (1, 64, math.ceil(h / 2), math.ceil(w / 2)), y.shape
    return "45x60 -> 23x30, 23x30 -> 12x15"


# ------------------------------------------------------------------ whole models
@check
def every_model_shapes_aux_and_gradients():
    for name in MODEL_ZOO:
        m = build_model(name).train()
        out = m(torch.randn(2, 3, 96, 128))
        loss_terms = list(out) if isinstance(out, tuple) else [out]
        if has_aux(name):
            assert len(loss_terms) == 2 and loss_terms[1].shape == (2, 11, 96, 128), name
        else:
            assert len(loss_terms) == 1, name
        assert loss_terms[0].shape == (2, 11, 96, 128), name
        sum(t.mean() for t in loss_terms).backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, (name, missing[:3])
        with torch.no_grad():
            y = m.eval()(torch.randn(1, 3, 90, 117))
        assert y.shape == (1, 11, 90, 117), (name, y.shape)
    return f"{len(MODEL_ZOO)} models; all parameters receive gradients; 90x117 input ok"


@check
def unet_parameter_count_matches_closed_form():
    def closed_form(b, nc=11):
        dc = lambda i, o: 9 * i * o + 2 * o + 9 * o * o + 2 * o
        ch = [b, 2 * b, 4 * b, 8 * b]
        p = dc(3, b) + sum(dc(ch[i], ch[i + 1]) for i in range(3))
        p += sum(ch[i + 1] * ch[i] * 4 + ch[i] + dc(2 * ch[i], ch[i]) for i in range(3))
        return p + b * nc + nc
    got = {b: count_params(build_model(n)) for b, n in [(32, "unet"), (20, "unet_pm")]}
    assert all(got[b] == closed_form(b) for b in got), (got, {b: closed_form(b) for b in got})
    return f"unet {got[32]:,}, unet_pm {got[20]:,}"


@check
def lmiinet_parameter_budget():
    n = count_params(build_model("lmiinet"))
    assert 0.65e6 < n < 0.80e6, n
    return f"{n:,} parameters (paper: 0.72M)"


@check
def fp16_forward_is_finite_and_close_to_fp32():
    if not torch.cuda.is_available():
        return "skipped (no CUDA)"
    torch.manual_seed(0)
    m = build_model("lmiinet").cuda().eval()
    x = torch.randn(2, 3, 360, 480, device="cuda")
    with torch.no_grad():
        ref = m(x)
        with torch.autocast("cuda"):
            y = m(x)
    agree = (y.argmax(1) == ref.argmax(1)).float().mean().item()
    assert torch.isfinite(y).all() and agree > 0.99, agree
    return f"argmax agreement fp16 vs fp32 {agree * 100:.2f}%"


@check
def cuda_graph_training_step_matches_eager_step():
    """Six Adam steps replayed from a captured CUDA graph agree with six eager steps.

    Two eager runs are not bitwise identical on GPU (atomic adds in bilinear-upsample backward and
    in the weighted loss), and Adam turns tiny gradient noise into full-size steps, so the graph run
    is compared against that measured eager-vs-eager floor, plus the per-step losses.
    """
    if not torch.cuda.is_available():
        return "skipped (no CUDA)"
    from .engine import GraphTrainer
    bench = torch.backends.cudnn.benchmark
    try:
        crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        g = torch.Generator().manual_seed(1)
        batches = [(torch.randn(2, 3, 96, 128, generator=g).cuda(),
                    torch.randint(0, 12, (2, 96, 128), generator=g).cuda()) for _ in range(6)]

        def run(use_graph):
            torch.manual_seed(0)
            m = build_model("lmiinet").cuda()
            m.head.drop = nn.Identity()                  # dropout RNG streams differ by design
            t = GraphTrainer(m, crit, m.parameters(), 1e-3, 2e-4, amp_dtype=None, use_graph=use_graph)
            losses = []
            for i, (x, y) in enumerate(batches):
                t.set_lr(poly_lr(1e-3, i, 6))
                t.step(x, y)
                losses.append(t.pop_mean_loss())
            return m, losses, t

        def param_diff(a, b):
            return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))

        e1, l1, _ = run(False)
        e2, _, _ = run(False)
        gm, lg, tg = run(True)
        assert tg.graph is not None, "graph was not captured"
        floor, d = param_diff(e1, e2), param_diff(e1, gm)
        loss_rel = max(abs(a - b) / abs(a) for a, b in zip(l1, lg))
        assert d <= 2 * floor + 1e-5 and loss_rel < 5e-3, (d, floor, loss_rel)
        return f"param diff graph {d:.1e} vs eager-eager floor {floor:.1e}; losses within {loss_rel:.1e}"
    finally:
        torch.backends.cudnn.benchmark = bench


# ------------------------------------------------------------------ data
@check
def augmentations_keep_labels_valid():
    from PIL import Image
    random.seed(0)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (360, 480, 3), dtype=np.uint8))
    lab = Image.fromarray(np.random.default_rng(1).integers(0, 12, (360, 480), dtype=np.uint8))
    allowed = set(range(12))
    for _ in range(20):
        i, m = RandomScale(0.75, 1.75)(img, lab)
        i, m = RandomCrop((360, 480))(i, m)
        i, m = RandomHFlip()(i, m)
        arr = np.array(m)
        assert i.size == (480, 360) and m.size == (480, 360)
        assert set(np.unique(arr)) <= allowed
    flipped = RandomHFlip(p=1.0)(img, lab)[1]
    assert np.array_equal(np.array(flipped), np.array(lab)[:, ::-1])
    return "scale/crop/flip: sizes fixed, no new label values, flip aligned"


@check
def gpu_augmentation_identity_flip_and_labels():
    """The batched affine warp reproduces the image exactly at scale 1, mirrors it under a flip,
    pads with void outside the source, and never invents label values."""
    from .data.gpu import affine_thetas, warp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(0)
    img = torch.randint(0, 256, (2, 3, 36, 48), dtype=torch.uint8, device=dev)
    lab = torch.randint(0, 11, (2, 36, 48), dtype=torch.uint8, device=dev)
    ident = affine_thetas(2, (36, 48), (36, 48), (1.0, 1.0), 0.0, g, dev)
    out, ol = warp(img, lab, ident, (36, 48))
    e_img = (out - img.float() / 255).abs().max().item()
    assert e_img < 1e-5 and torch.equal(ol, lab.long()), e_img
    flip = affine_thetas(2, (36, 48), (36, 48), (1.0, 1.0), 1.0, g, dev)
    out, ol = warp(img, lab, flip, (36, 48))
    assert (out - img.float().flip(-1) / 255).abs().max().item() < 1e-5 and torch.equal(ol, lab.long().flip(-1))
    small = affine_thetas(2, (36, 48), (36, 48), (0.5, 0.5), 0.0, g, dev)
    _, ol = warp(img, lab, small, (36, 48))
    assert (ol[:, 18:, :] == IGNORE_INDEX).all() and (ol[:, :, 24:] == IGNORE_INDEX).all()
    assert set(torch.unique(ol).tolist()) <= set(range(12))
    big = affine_thetas(64, (36, 48), (36, 48), (0.75, 1.75), 0.5, g, dev)
    _, ol = warp(img.repeat(32, 1, 1, 1), lab.repeat(32, 1, 1), big, (36, 48))
    assert set(torch.unique(ol).tolist()) <= set(range(12))
    return f"identity error {e_img:.1e}, flip exact, half-scale padded with void, labels valid"


@check
def camvid_split_sizes():
    root = "data/CamVid"
    if not os.path.isdir(root):
        return "skipped (data not downloaded yet)"
    from .data import CamVid
    sizes = {s: len(CamVid(root, s)) for s in ("train", "val", "test")}
    assert sizes == {"train": 367, "val": 101, "test": 233}, sizes
    _, mask, _ = CamVid(root, "train")[0]
    arr = np.array(mask)
    assert arr.shape == (360, 480) and arr.max() <= IGNORE_INDEX
    return "367 / 101 / 233, labels in 0..11"


def run_tests(out_csv="results/00_unit_tests.csv", log=print):
    rows, ok = [], True
    for fn in CHECKS:
        t0 = time.time()
        try:
            detail, status = fn(), "pass"
        except Exception as e:  # noqa: BLE001
            detail, status, ok = f"{type(e).__name__}: {e}", "FAIL", False
        rows.append([fn.__name__, status, detail, f"{time.time() - t0:.2f}"])
        log(f"  [{status}] {fn.__name__}: {detail}")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["check", "status", "detail", "seconds"])
        wr.writerows(rows)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
